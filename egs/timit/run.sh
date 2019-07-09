#!/bin/bash

source path.sh

CUDA_VISIBLE_DEVICES=0 python3 $ROOT/src/train.py \
    data \
    config_template.yaml \
    #--continue-training True 
