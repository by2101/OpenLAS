#!/bin/bash

source kaldi_path.sh

ref=decode_test/text
res=decode_test/result.txt
ali=aligment.txt


format_align=$KALDI_ROOT/egs/wsj/s5/utils/scoring/wer_per_utt_details.pl
align-text --special-symbol="'***'" ark:$ref ark:$res ark,t:- |\
    $format_align --special-symbol "'***'" > $ali
compute-wer --text --mode=present ark:$ref ark:$res 





