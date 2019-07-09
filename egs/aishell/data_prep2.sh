#!/usr/bin/env bash


data_root=/home/baiye/Speech/Speech-Transformer-torch-multigpu/egs/aishell/data

train_feats=${data_root}/train/feats.scp
train_trans=${data_root}/train/text
dev_feats=${data_root}/dev/feats.scp
dev_trans=${data_root}/dev/text
test_feats=${data_root}/test/feats.scp
test_trans=${data_root}/test/text

tmp=tmp
mkdir -p $tmp

. ./path.sh

mkdir -p data
python3 $ROOT/tools/add_space.py \
   $train_trans $tmp/train_text
python3 $ROOT/tools/add_space.py \
   $dev_trans $tmp/dev_text
python3 $ROOT/tools/add_space.py \
   $test_trans $tmp/test_text

