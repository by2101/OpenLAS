#!/bin/bash

source path.sh

data=data/test.json
vocab=data/vocab
model=exp3/model.pt
result=result.txt

CUDA_VISIBLE_DEVICES=1 python3 $ROOT/src/inference.py \
    $data \
    $vocab \
    $model 
