#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train
# @Author: Ye Bai
# @Date  : 2019/4/2
import sys
import os
import argparse
import logging
import yaml
import torch
from data import SpeechDataset, FrameBasedSampler, Collate, load_vocab
from utils import Timer, str2bool
from encoder import BiRNN_Torch, BiRNN
from decoder import RNNDecoder
from model import LAS
import schedule
from trainer import Trainer
import utils

import pdb

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# logging.basicConfig(format='train.py [line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

# logger = logging.getLogger()
# console = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('train.py [line:%(lineno)d] - %(levelname)s: %(message)s')
# console.setFormatter(formatter)
# logger.addHandler(console)

def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: train.py <data_file> <model_file> """)
    parser.add_argument("data_file", help="path to data file (json format)")
    parser.add_argument("vocab_file", help="path to vocab file")
    parser.add_argument("model_file", help="path to model file")
    parser.add_argument("result_file", help="path for writing decoding results.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    total_timer = Timer()
    total_timer.tic()
    
    args = get_args()
    
    pkg = torch.load(args.model_file)
    model_config = pkg['model_config']
    vocab = load_vocab(args.vocab_file)
    id2token = [None] * len(vocab)
    for k, v in vocab.items():
        id2token[v] = k
   
    collate = Collate(model_config["left_context"],
                      model_config["right_context"],
                      model_config["skip_frame"],
                      model_config["norm_mean"],
                      model_config["norm_var"])


    testset = SpeechDataset(args.data_file)
    test_loader = torch.utils.data.DataLoader(testset, collate_fn=collate, shuffle=False)

    # check dim match
    if model_config["feat_dim"] != testset[0]["feat"]["dim"]:
        raise ValueError(("Dim mismatch: "+
            "model {} vs. feat {}.").format(model_config["feat_dim"], testset[0]["feat"]["dim"]))
            
            
    model_load_timer = Timer()
    model_load_timer.tic()
    # build encoder and decoder
    if model_config["encoder"]["type"] == "BiRNN":
        encoder = BiRNN(model_config["encoder"])
    elif model_config["encoder"]["type"] == "BiRNN_Torch":
        encoder = BiRNN_Torch(model_config["encoder"])
    else:
        raise ValueError("Unknown encoder type.")

    if model_config["decoder"]["type"] == "RNNDecoder":
        decoder = RNNDecoder(model_config["decoder"])
    else:
        raise ValueError("Unknown decoder type.")
        
    model = LAS(encoder, decoder, model_config)
    model.load_state_dict(pkg['state_dict'])
    model = model.cuda()
    model.eval()
    
    logger.info("Spend {:.3f} sec for building model..".format(model_load_timer.toc()))
    model_load_timer.tic()
    # model = model.cuda()
    logger.info("Spend {:.3f} sec for loading model to gpu..".format(model_load_timer.toc()))
    logger.info("Model: \n{}".format(model))
    
    tot_utt = len(testset)
    decode_cnt = 0
    fw = open(args.result_file, 'w', encoding="utf8")
    # batch_size is 1, only one sentence.
    for utts, feats, feat_lengths, src_ids, src_lengths, tgt_ids, tgt_lengths in test_loader:
        feats = feats.cuda()
        feat_lengths = feat_lengths.cuda()
        decode_cnt += 1
        logger.info("Decoding {} [{}/{}]\n".format(utts[0], decode_cnt, tot_utt))
        ref = " ".join([id2token[id] for id in tgt_ids.view(-1)])
        best_hyp, ended_hyps = model.beam_search_sentence(feats, feat_lengths, id2token)
        hyp = " ".join([id2token[id] for id in best_hyp['ids']])
        logger.info("\nref:\t{}\nhyp:\t{}\n".format(ref, hyp))
        res = "{} {}\n".format(utts[0], " ".join([id2token[id] for id in best_hyp['ids'][:-1]]))
        fw.write(res)
        
    fw.close()    
        
    logger.info("Finished. Total Time: {:.3f} hrs.".format(total_timer.toc()/3600.))












