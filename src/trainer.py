#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train
# @Author: Ye Bai
# @Date  : 2019/3/30
import os
import sys
import logging
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init
import utils
from utils import Timer


    

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)

# def stat_grad(named_parameters):
    # if isinstance(named_parameters, torch.Tensor):
        # named_parameters = [named_parameters]
    # parameters = list(filter(lambda p: p[1].grad is not None, named_parameters))

    # stats = []
    # with torch.no_grad():
        # for p in parameters:
            # # print("Grad: ", p[0], p[1].grad)
            # max = p[1].grad.max().item()
            # min = p[1].grad.min().item()
            # mean = torch.mean(p[1].grad).item()
            # var = torch.mean((p[1].grad-mean)**2.).item()
            # stats.append((p[0],max, min, mean, var))

    # return stats

# def stats_to_string(stats):
    # msg = ""
    # for stat in stats:
        # msg += (("{} Max: {:.10f} Min: {:.10f} " +
              # "Mean: {:.10f} Var:{:.10f}\n").format(stat[0], stat[1], stat[2], stat[3], stat[4]))
    # return msg

# def weights_init(m):
    # for k, w in m.state_dict().items():
        # if k.find('rnn') != -1:
            # if k.find('weight') != -1:
                # init.uniform_(w, -0.1, +0.1)
                # logger.info("Init RNN weight with uniform between [-0.1, 0.1]")
            # elif k.find('bias') != -1:
                # init.constant_(w, 1.0)
                # logger.info("Init RNN bias at 1.0")
        # elif k.find("affine") != -1:
            # if k.find('weight') != -1:
                # init.xavier_normal_(w)
                # init.uniform_(w, -0.1, +0.1)
                # # logger.info("Init Affine with xavier normal")
                # logger.info("Init Affine with uniform between [-0.1, 0.1]")
            # elif k.find('bias') != -1:
                # init.constant_(w, 1.0)
                # logger.info("Init Affine bias to 0.0")


# class PreFetcher(object):
    # def __init__(self, loader):
        # self.loader = loader:
        # self.next_data = None
    
    # def preload(self):
        # try:
            
    

class Trainer(object):
    def __init__(self, optimizer, scheduler, tr_loader, model, config, cv_loader=None):
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader
        self.model = model
        if "multi_gpu" in config and config["multi_gpu"] == True:
            self.model_to_pack = self.model.module
        self.num_epoch = config["num_epoch"]
        self.exp_dir = config["exp_dir"]
        self.print_inteval = config["print_inteval"]
        self.init_lr = config["init_lr"]
        self.optimizer = optimizer      
        self.scheduler = scheduler
        if "label_smooth" in config:
            self.label_smooth = config["label_smooth"] 
        else:
            self.label_smooth = 0.

        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

        # training hyper-parameters
        self.grad_max_norm = config["grad_max_norm"]

        # trainer state
        self.epoch = 0
        self.step = 0
        self.tr_loss = []
        self.cv_loss = []

    def train(self):
        train_timer = Timer()
        best_cv_loss = 9e9
        if self.cv_loss:
            best_cv_loss = self.cv_loss[-1]

            
        while self.epoch < self.num_epoch:
            self.epoch += 1
            train_timer.tic()
            epoch_avg_loss = self.iter_one_epoch()
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Training Summmary: avg loss {}\n".format(self.epoch, epoch_avg_loss)
            msg += "-"*85
            logger.info(msg)
            self.tr_loss.append(epoch_avg_loss)
            if utils.TENSORBOARD_LOGGING == 1:
                utils.visulizer.add_scalar("tr_loss", epoch_avg_loss, self.epoch)

            
            if self.cv_loader is not None:
                epoch_avg_loss = self.iter_one_epoch(cross_valid=True)
                msg = "\n" + "-"*85 + "\n"
                msg += ("Epoch {} Valid Summmary: avg loss {}\n".format(self.epoch, epoch_avg_loss))
                msg += "-"*85
                logger.info(msg)
                self.cv_loss.append(epoch_avg_loss)
                this_cv_loss = epoch_avg_loss
                if best_cv_loss > this_cv_loss:
                    logger.info("Find a better model!")
                    self.save(os.path.join(self.exp_dir, "model.pt"))
                    best_cv_loss = this_cv_loss
                if utils.TENSORBOARD_LOGGING == 1:
                    utils.visulizer.add_scalar("cv_loss", this_cv_loss, self.epoch)


            self.save(os.path.join(self.exp_dir, "last_model.pt"))
            time_cost = train_timer.toc()
            logger.info("Time cost: {:.1f}s\n".format(time_cost) + "-"*85)
            # if utils.TENSORBOARD_LOGGING == 1:
                # utils.visulizer.close()  


    def iter_one_epoch(self, cross_valid=False):
        iter = 0
        epoch_timer = Timer()
        epoch_timer.tic()

        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
        else:
            loader = self.tr_loader
            self.model.train()

        tot_loss = 0.
        utt_itered = 0.
 
        for utts, feats, feat_lengths, src_ids, src_lengths, tgt_ids, tgt_lengths in loader:
            iter += 1
            if not cross_valid:
                self.step += 1
                
            feats = feats.cuda()
            feat_lengths = feat_lengths.cuda()
            src_ids = src_ids.cuda()
            src_lengths = src_lengths.cuda()
            tgt_ids = tgt_ids.cuda()
            tgt_lengths = tgt_lengths.cuda()
            
            if not cross_valid:
                loss = self.model(feats, feat_lengths, src_ids, tgt_ids, self.label_smooth)
            else:
                loss = self.model(feats, feat_lengths, src_ids, tgt_ids)
            
            loss = torch.mean(loss)
            
            
            if not cross_valid:
                self.scheduler.step()
                self.scheduler.set_lr(self.optimizer, self.init_lr)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
                self.optimizer.step()
                if utils.TENSORBOARD_LOGGING == 1:
                    utils.visulizer.add_scalar("learning_rate", 
                        list(self.optimizer.param_groups)[0]["lr"], self.step)
                    utils.visulizer.add_scalar("training_loss", 
                        loss.item(), self.step)
                

            tot_loss += loss.item()
            utt_itered += int(feat_lengths.shape[0])
                
                
            cost_time = epoch_timer.toc()

            if self.step % self.print_inteval == 0 and not cross_valid:
                msg = ("Epoch {} | Step {} | Iter {}:\n" +
                        "Curr Loss: {:.4f} | Lr: {:.6f} "+
                        "Itered Sentences: {} Speed: {:.4f} secs/sentence").format(
                        self.epoch, self.step, iter, 
                        loss.item(), list(self.optimizer.param_groups)[0]["lr"], 
                        utt_itered, cost_time/utt_itered)
                logger.info("Progress:\n" + msg.strip())              
                if utils.TENSORBOARD_LOGGING == 1:
                    f, fl, si = feats[0][None, :, :], feat_lengths[0][None], src_ids[0][None, :]
                    att_list = self.model_to_pack.get_atten_scores(f, fl, si)              
                    for i, att_scores in enumerate(att_list):
                        utils.visulizer.add_img_figure("att_img_{}".format(i), 
                            att_scores[0].detach().cpu().numpy())
                        
                
        return tot_loss/iter

    def package(self):
        schedule_pkg = self.scheduler.pack_state()  
        pack = {
            # state
            'model_config': self.model_to_pack.config,
            'encoder_config': self.model_to_pack.encoder.config,
            'decoder_config': self.model_to_pack.decoder.config,
            'state_dict': self.model_to_pack.state_dict(),
            'init_lr': self.init_lr,
            'schedule_pkg': schedule_pkg,
            'optim_state': self.optimizer.state_dict(),
            # progress
            'epoch': self.epoch,
            'step': self.step,
            'lr': list(self.optimizer.param_groups)[0]["lr"],
            'tr_loss': self.tr_loss,
        }
        if self.cv_loss:
            pack['cv_loss'] = self.cv_loss
        return pack

    def save(self, path):
        pkg = self.package()
        torch.save(pkg, path)
        logger.info("Saving model to {}".format(path))

    def restore_trainer_state(self, package):
        self.epoch = package['epoch']
        self.step = package['step']
        self.tr_loss = package['tr_loss']
        if 'cv_loss' in package:
            self.cv_loss = package['cv_loss']


# For debug
if __name__ == "__main__":
    class Config:
        input_dim = 200
        hidden_size = 128
        nlayers = 2
        num_targets = -1


    class TrainConfig:
        num_epoch = 100000

    vocab = load_vocab("/data1/baiye/Speech/pytorch-ctc/egs/timit/data/vocab")
    fn = "/data1/baiye/Speech/pytorch-ctc/egs/timit/data/train.json"
    dataset = SpeechDataset(fn)
    collate = Collate(2, 2, 1, True, False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate, shuffle=True)

    config = Config()
    config.num_targets = len(vocab)
    rnn_ctc = RNN(config)
    logger.info("\n{}".format(rnn_ctc))



    # pdb.set_trace()
    weights_init(rnn_ctc)
    rnn_ctc = rnn_ctc.cuda()

    optimizer = torch.optim.SGD(rnn_ctc.parameters(), lr=1e-3, momentum=0.9)

    train_config = TrainConfig()


    trainer = Trainer(dataloader, rnn_ctc, optimizer, train_config)
    trainer.train()



