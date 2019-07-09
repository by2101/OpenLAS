#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train
# @Author: Ye Bai
# @Date  : 2019/4/2
import sys
import os
import argparse
import logging
import yaml
import torch
from data import SpeechDataset, FrameBasedSampler, Collate, load_vocab
from utils import Timer, str2bool
from encoder import BiRNN_Torch, BiRNN
from decoder import RNNDecoder
from model import LAS
import schedule
from trainer import Trainer
import utils


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# logging.basicConfig(format='train.py [line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

# logger = logging.getLogger()
# console = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('train.py [line:%(lineno)d] - %(levelname)s: %(message)s')
# console.setFormatter(formatter)
# logger.addHandler(console)

def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: train.py <data_dir> <config>""")
    parser.add_argument("data_dir", help="data directory")
    parser.add_argument("config", help="path to config file")
    parser.add_argument('--continue-training', type=str2bool, default=False,
                        help='Continue training from last_model.pt.')
    args = parser.parse_args()
    return args

def derive_model_config(model_config, vocab_size):
    input_dim = (model_config["feat_dim"] * 
        (model_config["left_context"]+model_config["right_context"]+1))
    model_config["encoder"]["input_dim"] = input_dim
    model_config["decoder"]["vocab_size"] = vocab_size
    return model_config
    
    
def check_data_dir(data_dir):
    for fn in ["vocab", "train.json"]:
        if not os.path.exists(os.path.join(data_dir, fn)):
            raise ValueError("file {} does not exist.".format(fn))

if __name__ == "__main__":

    total_timer = Timer()
    total_timer.tic()
    x = torch.zeros(2)
    x.cuda() # for initialize gpu
    logger.info("Spend {:.3f} sec for x.cuda()..".format(total_timer.toc()))
    
    args = get_args()
    check_data_dir(args.data_dir)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer_config = config["trainer"]
    model_config = config["model"]
    schedule_config = config["schedule"]
    vocab = load_vocab(os.path.join(args.data_dir, "vocab"))

    model_config = derive_model_config(model_config, len(vocab))
    
    if utils.TENSORBOARD_LOGGING == 1:
        utils.visulizer.set_writer(os.path.join(trainer_config["exp_dir"], 'log'))    
    
    collate = Collate(model_config["left_context"],
                      model_config["right_context"],
                      model_config["skip_frame"],
                      model_config["norm_mean"],
                      model_config["norm_var"])
                      
    batch_frames = trainer_config["batch_frames"]
    valid_batch_size = 20
    if "multi_gpu" in trainer_config and trainer_config["multi_gpu"] == True:
        batch_frames *= torch.cuda.device_count()
        valid_batch_size *= torch.cuda.device_count()
    

    trainset = SpeechDataset(os.path.join(args.data_dir, "train.json"))
    validset = SpeechDataset(os.path.join(args.data_dir, "dev.json"))
    logger.info("Loaded {} utterances for training.".format(len(trainset)))
    logger.info("Loaded {} utterances for validation.".format(len(validset)))
    
    
    trainsampler = FrameBasedSampler(trainset, frame_num=batch_frames)
    tr_loader = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler, 
        collate_fn=collate, shuffle=False, num_workers=16, pin_memory=True)
    cv_loader = torch.utils.data.DataLoader(validset, 
        collate_fn=collate, batch_size=valid_batch_size, num_workers=16, shuffle=False, pin_memory=True)

    # check dim match
    if model_config["feat_dim"] != trainset[0]["feat"]["dim"]:
        raise ValueError(("Dim mismatch: "+
            "model {} vs. feat {}.").format(model_config["feat_dim"], trainset[0]["feat"]["dim"]))
            
    if args.continue_training:
        pkg = torch.load(os.path.join(trainer_config["exp_dir"], 'last_model.pt'))
        model_config_restored = pkg['model_config']

        # check
        for key in model_config_restored.keys():
            if key not in model_config:
                raise ValueError(("The restored model does not match the config. "+
                                 "Please check it. {} is not in model_config.").format(key))
            if model_config_restored[key] != model_config[key]:
                raise ValueError(("The restored model does not match the config. "+
                                "Please check it. \n"+
                                "restored {} is {}, config is {}.").format(key,
                                model_config_restored[key], model_config[key]))
        model_config = model_config_restored            
            

    model_load_timer = Timer()
    model_load_timer.tic()
    
    # build encoder and decoder
    if model_config["encoder"]["type"] == "BiRNN":
        encoder = BiRNN(model_config["encoder"])
    elif model_config["encoder"]["type"] == "BiRNN_Torch":
        encoder = BiRNN_Torch(model_config["encoder"])
    else:
        raise ValueError("Unknown encoder type.")

    if model_config["decoder"]["type"] == "RNNDecoder":
        decoder = RNNDecoder(model_config["decoder"])
    else:
        raise ValueError("Unknown decoder type.")        
    
    model = LAS(encoder, decoder, model_config)
    
    if args.continue_training:
        model.load_state_dict(pkg['state_dict'])
    
    logger.info("Spend {:.3f} sec for building model..".format(model_load_timer.toc()))
    model_load_timer.tic()
    
    if "multi_gpu" in trainer_config and trainer_config["multi_gpu"] == True:
        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    
    model = model.cuda()
    logger.info("Spend {:.3f} sec for loading model to gpu..".format(model_load_timer.toc()))
    
    if trainer_config["optim_type"] == "sgd": 
        optimizer = torch.optim.SGD(model.parameters(), lr=trainer_config["init_lr"], momentum=0.9)
    elif trainer_config["optim_type"] == "adam": 
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config["init_lr"], 
            betas=(0.9, 0.98), eps=1e-09)
    else:
        raise ValueError("Unknown optimizer type.")       

    
    if args.continue_training:
        optimizer.load_state_dict(pkg['optim_state'])
    
    if schedule_config["type"] == "linear_decay":
        scheduler = schedule.LinearLearningRateSchedule(
                schedule_config["x0"],
                schedule_config["y0"],
                schedule_config["x1"],
                schedule_config["y1"]
            )
    elif schedule_config["type"] == "linear_decay_warmup":
        scheduler = schedule.WarmupLinearLearningRateSchedule(
                schedule_config["warmup_steps"],
                schedule_config["x0"],
                schedule_config["y0"],
                schedule_config["x1"],
                schedule_config["y1"]
            )        
    else:
        raise ValueError("Unknown scheduler type.")

    if args.continue_training:
        scheduler.restore_state(pkg['schedule_pkg'])        
        
    logger.info("Model: \n{}".format(model))
    if args.continue_training:
        logger.info("Restore Training...")
    else:
        logger.info("Start Training...")
    trainer = Trainer(optimizer, scheduler, tr_loader, model, trainer_config, cv_loader)
    if args.continue_training:
        trainer.restore_trainer_state(pkg)
    trainer.train()
    logger.info("Total Time: {:.1f} hrs.".format(total_timer.toc()/3600.))












