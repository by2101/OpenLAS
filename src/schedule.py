import torch
import math

class BaseLearningRateSchedule(object):
    
    def __init__(self):
        self.step_num = 0
        self.decay_rate = 1.
        
    def set_lr(self, optimizer, init_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr * self.decay_rate

    def step(self):
        self.step_num += 1
        self.update_decay_rate()    
            
    def pack_state(self):
        pkg = {
            "step": self.step_num,
            "decay_rate": self.decay_rate,
            }
        return pkg
        
    def restore_state(self, pkg):
        self.step_num = pkg['step']
        self.decay_rate = pkg['decay_rate']
     
    def update_decay_rate(self):
        raise NotImplementedError()
    
    
def compute_polynomial_intep(x, x0, y0, x1, y1, power):
    if x < x0:
        return y0
    elif x > x1:
        return y1
    else:
        if power != 1.0:
            f = ((1.0 * x - x0) / (x1 - x0)) ** power      
        else:
            f = ((1.0 * x - x0) / (x1 - x0))
        y = y0 + f * (y1 - y0)
        return y
        
def compute_linear_intep(x, x0, y0, x1, y1):
    return compute_polynomial_intep(x, x0, y0, x1, y1, 1.0)
    
    
class LinearLearningRateSchedule(BaseLearningRateSchedule):    
    def __init__(self, x0, y0, x1, y1):
        super(LinearLearningRateSchedule, self).__init__()
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        
    def pack_state(self):
        pkg = {
            "step": self.step_num,
            "decay_rate": self.decay_rate,
            "x0": self.x0,
            "x1": self.x1,
            "y0": self.y0,
            "y1": self.y1,
            }
        return pkg
        
    def restore_state(self, pkg):
        self.step_num = pkg['step']
        self.decay_rate = pkg['decay_rate']
        self.x0 = pkg['x0']
        self.x1 = pkg['x1']
        self.y0 = pkg['y0']
        self.y1 = pkg['y1']
     
    def update_decay_rate(self):
        self.decay_rate = compute_linear_intep(self.step_num, self.x0, 
            self.y0, self.x1, self.y1)    

         
class WarmupLinearLearningRateSchedule(BaseLearningRateSchedule):    
    def __init__(self, warmup_step, x0, y0, x1, y1):
        super(WarmupLinearLearningRateSchedule, self).__init__()
        self.warmup_step = warmup_step
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        
    def pack_state(self):
        pkg = {
            "step": self.step_num,
            "decay_rate": self.decay_rate,
            "warmup_step": self.warmup_step,
            "x0": self.x0,
            "x1": self.x1,
            "y0": self.y0,
            "y1": self.y1,
            }
        return pkg
        
    def restore_state(self, pkg):
        self.step_num = pkg['step']
        self.decay_rate = pkg['decay_rate']
        self.warmup_step = pkg['warmup_step']
        self.x0 = pkg['x0']
        self.x1 = pkg['x1']
        self.y0 = pkg['y0']
        self.y1 = pkg['y1']
     
    def update_decay_rate(self):
        dc0 = compute_linear_intep(self.step_num, 0, 
            0, self.warmup_step, self.y0)
        dc1 = compute_linear_intep(self.step_num, self.x0, 
            self.y0, self.x1, self.y1)
        self.decay_rate = min(dc0, dc1)  