#!/bin/bash

source path.sh

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 $ROOT/src/train.py \
    data \
    config_myrnn_ls.yaml \
    #--continue-training True 
