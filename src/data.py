#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data.py
# @Author: Ye Bai
# @Date  : 2019/3/30
import json
import logging
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import kaldi_io as kio

BOS_SYM="<bos>"
BOS_ID=0
EOS_SYM="<eos>"
EOS_ID=1
UNK_SYM="<unk>"
UNK_ID=2
IGNORE_ID=-1

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("data")
logger.setLevel(logging.INFO)

def load_vocab(fn):
    dic = {}
    with open(fn) as f:
        for line in f:
            items = line.strip().split()
            dic[items[0]] = int(items[1])
    if BOS_SYM not in dic or dic[BOS_SYM] != BOS_ID:
        raise ValueError("{} is not {}.".format(BOS_SYM, BOS_ID))
    if EOS_SYM not in dic or dic[EOS_SYM] != EOS_ID:
        raise ValueError("{} is not {}.".format(EOS_SYM, UNK_ID))
    if UNK_SYM not in dic or dic[UNK_SYM] != UNK_ID:
        raise ValueError("{} is not {}.".format(UNK_SYM, UNK_ID))        
    return dic

class SpeechDataset(data.Dataset):

    def __init__(self, data_json_path, sort_by_length=True):
        super(SpeechDataset, self).__init__()
        with open(data_json_path, 'rb') as f:
            data = json.load(f)

        if sort_by_length:
            self.data = sorted(data, key=lambda x:x["trans"]["length"])
        else:
            self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def norm_feats(feats, norm_mean=True, norm_var=True):
    if norm_mean == False and norm_var == False:
        return feats
    mean = np.mean(feats, axis=0)
    var = np.mean((feats-mean)**2.)
    new_feats = feats.copy()
    if norm_mean:
        new_feats -= mean
    if norm_var:
        new_feats /= np.sqrt(var+1e-6)
    return new_feats

def compute_frame(T, left, skip):
    return (T + left) // skip

def splice_feat(feat, left, right, skip):
    T, C = feat.shape
    new_feat = []

    padded_feat = []
    if left >= 1:
        padded_feat.append(np.stack([feat[0]] * left))
    padded_feat.append(feat)
    if right >= 1:
        padded_feat.append(np.stack([feat[-1]] * right))

    padded_feat = np.concatenate(padded_feat)
    for t in range(left, left+T, skip):
        new_feat.append(padded_feat[t-left:t+right+1].flatten())
    new_feat = np.vstack(new_feat)
    return new_feat

def load_examples(data, left, right, skip, norm_mean, norm_var):
    max_feat_length = max([compute_frame(d["feat"]["length"], left, skip) for d in data])
    max_src_length = max([d["trans"]["length"] for d in data]) + 1 # for BOS and EOS
    max_tgt_length = max([d["trans"]["length"] for d in data]) + 1 # for BOS and EOS

    utts = []
    feats = []
    feat_lengths = []
    src_ids = []
    src_lengths = []
    tgt_ids = []
    tgt_lengths = []

    for d in data:
        utts.append(d["utt"])
        feat = kio.read_mat(d["feat"]["path"])
        feat = norm_feats(feat, norm_mean=norm_mean, norm_var=norm_var)
        padded_feat = np.zeros([max_feat_length, d["feat"]["dim"]*(left+1+right)])
        new_feat = splice_feat(feat, left, right, skip)
        padded_feat[:len(new_feat)] += new_feat
                
        ids = [BOS_ID] + d["trans"]["tokenids"] + [EOS_ID]
        src = ids[:-1]
        tgt = ids[1:]
        src_lengths.append(len(src))
        tgt_lengths.append(len(tgt))
        
        src_padded = src + [EOS_ID] * (max_src_length-len(src))
        tgt_padded = tgt + [IGNORE_ID] * (max_tgt_length-len(tgt))              

        feats.append(padded_feat)
        feat_lengths.append(len(new_feat))
        
        src_ids.append(src_padded)
        tgt_ids.append(tgt_padded)
        

    feats = torch.Tensor(np.stack(feats))
    feat_lengths = torch.Tensor(np.array(feat_lengths)).int()
    src_ids = torch.Tensor(np.stack(src_ids)).long()
    src_lengths = torch.Tensor(np.array(src_lengths)).int()
    tgt_ids = torch.Tensor(np.stack(tgt_ids)).long()
    tgt_lengths = torch.Tensor(np.array(tgt_lengths)).int()    
    return utts, feats, feat_lengths, src_ids, src_lengths, tgt_ids, tgt_lengths

class Collate(object):
    def __init__(self, left=1, right=1, skip=1, norm_mean=False, norm_var=False):
        self.left = left
        self.right = right
        self.skip = skip
        self.norm_mean = norm_mean
        self.norm_var = norm_var

    def __call__(self, batch):
        return load_examples(batch, self.left, self.right, self.skip, self.norm_mean, self.norm_var)

        
class FrameBasedSampler(Sampler):
    
    def __init__(self, dataset, frame_num=20000):
        self.dataset = dataset
        self.frame_num = frame_num
        
        batchs = []
        batch = []
        nframe = 0
        for idx in range(len(self.dataset)):
            batch.append(idx)
            nframe += self.dataset[idx]["feat"]["length"] 
            if nframe >= self.frame_num:
                batchs.append(batch)
                batch = []
                nframe = 0
        if batch:
            batchs.append(batch)
        self.batchs = batchs           
            
    def __iter__(self):
        np.random.shuffle(self.batchs)
        for b in self.batchs:
            yield b
        
        
    def __len__(self):
        return len(self.batchs)
        
        
        
class KaldiFeatLoader(object):
    def __init__(self, fn_scp, left, right, skip, norm_mean, norm_var):   
        self.fn_scp = fn_scp
        self.left = left
        self.right = right
        self.skip = skip
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        
    def iter(self):
        for key, feat in kio.read_mat_scp(self.fn_scp):
            feat = norm_feats(feat, norm_mean=self.norm_mean, norm_var=self.norm_var)
            feat = splice_feat(feat, self.left, self.right, self.skip)
            yield key, feat
        
# for debug
if __name__ == "__main__":
    fn = "/home/baiye/Speech/las/egs/timit/data/test.json"
    dataset = SpeechDataset(fn)
    sampler = FrameBasedSampler(dataset)
    collate = Collate(left=0, right=0)
    dataloader = data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, shuffle=False)
    dataiter = iter(dataloader)
    batch = next(dataiter)
    ori_feat = batch[0]
    _, idx_sort = torch.sort(batch[1], dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    x = batch[0].index_select(0, idx_sort)
    lengths = batch[1].index_select(0, idx_sort)
    x_packed = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
    x_padded = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
    x_padded = x_padded[0].index_select(0, idx_unsort)
















