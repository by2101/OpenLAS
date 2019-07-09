#!/bin/bash

source path.sh

data=data/test.json
vocab=data/vocab
model=exp_myrnn/model.pt
result=exp_myrnn/test_result.txt

CUDA_VISIBLE_DEVICES=0 python3 $ROOT/src/inference.py \
    $data \
    $vocab \
    $model \
    $result
