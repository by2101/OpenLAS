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

echo "<bos> 0" > data/vocab
echo "<eos> 1" >> data/vocab
echo "<unk> 2" >> data/vocab

python3 $ROOT/tools/add_space.py \
   $train_trans $tmp/train_text
python3 $ROOT/tools/add_space.py \
   $dev_trans $tmp/dev_text
python3 $ROOT/tools/add_space.py \
   $test_trans $tmp/test_text

cut -d" " -f2- $tmp/train_text |\
  tr ' ' '\n' | sort -u | sed '/^[[:space:]]*$/d' > $tmp/wordlist

cat $tmp/wordlist |\
    awk 'BEGIN{i=3; print "sil "i}{i+=1; print $1" "i}' >> data/vocab

python3 $ROOT/tools/prepare_data.py \
    data/vocab \
    $train_feats \
    $tmp/train_text \
    data/train.json

python3 $ROOT/tools/prepare_data.py \
    data/vocab \
    $dev_feats \
    $tmp/dev_text \
    data/dev.json

python3 $ROOT/tools/prepare_data.py \
    data/vocab \
    $test_feats \
    $tmp/test_text \
    data/test.json
