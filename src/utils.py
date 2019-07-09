#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: Ye Bai
# @Date  : 2019/3/30
import argparse
import sys
import time
import logging
import torch
from torch.nn import Parameter
import numpy as np


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

TENSORBOARD_LOGGING = 1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
 
 
class Timer(object):
    def __init__(self):
        self.start = 0.

    def tic(self):
        self.start = time.time()

    def toc(self):
        return time.time() - self.start

        
def get_seq_mask(inputs, lengths):
    with torch.no_grad():
        masks = torch.zeros_like(inputs)
        for i in range(lengths.shape[0]):
            masks[i, :lengths[i].long(), :] = 1
    return masks

    
def get_seq_mask_by_shape(max_length, dim, lengths):
    b = lengths.size(0)
    with torch.no_grad():    
        masks = torch.zeros(b, max_length, dim)
        for i in range(b):
            masks[i, :lengths[i].long(), :] = 1
    return masks
            
class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


if TENSORBOARD_LOGGING == 1:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboardX import SummaryWriter
    
    FIGURE_INTEVEL = 10
    
    class Visulizer(object):
        def __init__(self):
            self.writer = None
            self.fig_step = 0
        
        def set_writer(self, log_dir):
            if self.writer is not None:
                raise ValueError("Dont set writer twice.")
            self.writer = SummaryWriter(log_dir)
                
        def add_scalar(self, tag, value, step):
            self.writer.add_scalar(tag=tag, 
                scalar_value=value, global_step=step)
                
        def add_graph(self, model):
            self.writer.add_graph(model)

        def add_image(self, tag, img, data_formats):
            self.writer.add_image(tag, 
                img, 0, dataformats=data_formats)
                
        def add_img_figure(self, tag, img, step=None):
            fig, axes = plt.subplots(1,1)
            axes.imshow(img)
            self.writer.add_figure(tag, fig, global_step=step)
                
        def close(self):
            self.writer.close()
            
    visulizer = Visulizer()

        