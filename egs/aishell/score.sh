#!/bin/bash

tool=/home/baiye/Speech/kaldi/tools/sctk-2.4.10/bin/sclite
source ./path.sh

expdir=exp_myrnn
trans_src=tmp/test_text
res_src=$expdir/test_result.txt

python3 $ROOT/tools/format_text_for_sclite.py \
  $trans_src $expdir/trans.trn

python3 $ROOT/tools/format_text_for_sclite.py \
  $res_src $expdir/res.trn

$tool -i wsj -r $expdir/trans.trn -h $expdir/res.trn -e utf-8 -o all -O $expdir -c NOASCII





