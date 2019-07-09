#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : prepare_data.py
# @Author: Ye Bai
# @Date  : 2019/3/30

import argparse
import numpy as np
import logging
import json
from data import load_vocab, UNK_SYM
import kaldi_io as kio

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("prepare_data")
logger.setLevel(logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: extend_segment_times.py <vocab> <feats_scp> <trans_scp> <dest_json>""")
    parser.add_argument("vocab", help="path to vocab")
    parser.add_argument("feats_scp", help="path to feats.scp")
    parser.add_argument("trans_scp", help="path to trans.scp")
    parser.add_argument("dest_json", help="path to dest.json")
    args = parser.parse_args()
    return args

    
def parse_scp(fn):
    scp_dic = {}
    with open(fn) as f:
        for line in f:
            items = line.strip().split()
            scp_dic[items[0]] = " ".join(items[1:])
    return scp_dic


def trans_to_ids(trans, vocab):
    return [vocab[w] if w in vocab else vocab[UNK_SYM] for w in trans.strip().split()]


def get_feat_shape(feat_path):
    feat = kio.read_mat(feat_path)
    return feat.shape


def prep_data_json(feats_scp_dic, trans_scp_dic, vocab_dic, fnw):
    keys = feats_scp_dic.keys()
    examples = []
    for key in keys:
        if key not in trans_scp_dic:
            logger.warning("{} does not have transcripts, ignore it".format(key))
            continue
        eg = {}
        eg["utt"] = key
        length, dim = get_feat_shape(feats_scp_dic[key])
        eg["feat"] = {
            "length": length,
            "dim": dim,
            "path": feats_scp_dic[key],
            }
        trans_ids = trans_to_ids(trans_scp_dic[key], vocab_dic)
        eg["trans"] = {
            "tokens": trans_scp_dic[key],
            "tokenids": trans_ids,
            "length": len(trans_ids),
        }
        examples.append(eg)

    with open(fnw, 'w') as f:
        json.dump(examples, f)


if __name__ == "__main__":
    args = get_args()
    vocab = load_vocab(args.vocab)
    feats_scp = parse_scp(args.feats_scp)
    trans_scp = parse_scp(args.trans_scp)
    prep_data_json(feats_scp, trans_scp, vocab, args.dest_json)









