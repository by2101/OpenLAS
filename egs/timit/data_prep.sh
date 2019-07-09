#!/usr/bin/env bash


data_root=/home/baiye/Speech/kaldi/egs/timit/s5/data

raw_dict=$data_root/local/dict/lexicon.txt

train_feats=${data_root}/train/feats.scp
train_trans=${data_root}/train/text
dev_feats=${data_root}/dev/feats.scp
dev_trans=${data_root}/dev/text
test_feats=${data_root}/test/feats.scp
test_trans=${data_root}/test/text

. ./path.sh

mkdir -p data

echo "<bos> 0" > data/vocab
echo "<eos> 1" >> data/vocab
echo "<unk> 2" >> data/vocab
cat $raw_dict |\
    grep -v 'sil' |\
    awk 'BEGIN{i=3; print "sil "i}{i+=1; print $1" "i}' >> data/vocab

python3 $ROOT/tools/prepare_data.py \
    data/vocab \
    $train_feats \
    $train_trans \
    data/train.json

python3 $ROOT/tools/prepare_data.py \
    data/vocab \
    $dev_feats \
    $dev_trans \
    data/dev.json

python3 $ROOT/tools/prepare_data.py \
    data/vocab \
    $test_feats \
    $test_trans \
    data/test.json
