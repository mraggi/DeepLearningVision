import fastai.basics as fai
#import fastai.vision as fv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from delegation import delegates
from Layers import avg_conv_weight

import random

class LabelSmoothingCrossEntropyFixed(fai.Module):
    def __init__(self, eps:float=0.1):
        self.eps = eps

    def forward(self, output, target, reduction='mean'):
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        if reduction=='sum': 
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=reduction)

class WDModelWrapper(nn.Module):
    def __init__(self, model, model_loss, model_loss_multiplier = 1):
        super().__init__()
        self.model = model
        self.model_loss = model_loss
        self.mult = model_loss_multiplier
    
    def forward(self, x):
        if self.training:
            return self.model_loss(self.model)*self.mult, self.model(x)
        else:
            return self.model(x)

class WDLossWrapper(fai.Module):
    def __init__(self, loss_func):
        self.loss_func = loss_func

    def forward(self, y, targ, **kwargs):
        if fai.is_listy(y):
            #print(f"Shit-A: y={y}")
            model_loss, pred = y
            orig_loss = self.loss_func(pred, targ, **kwargs)
            
            return model_loss + orig_loss
        else:
            #print(f"Shit-B: y={y}")
            # We are in eval mode!
            return self.loss_func(y,targ, **kwargs)
    
def bn_loss(bn):
    x = bn.weight
    return torch.mean(x*x)

def idconvloss(c):
    T = c.weight
    s = c.stride[0]
    no,ni,k,k = T.shape
    T = T[:ni]
    Q = avg_conv_weight(ni,k,s,device=T.device)
    D = T-Q
    return torch.mean(D*D)

def model_loss(model):
    loss = 0.
    modules = fai.flatten_model(model)
    for m in modules:
        if isinstance(m, nn.Conv2d) and hasattr(m,'init_avg'):
            loss += idconvloss(m)*m.wd
        elif isinstance(m, nn.BatchNorm2d) and hasattr(m, 'init_zero'):
            loss += bn_loss(m)*m.wd
    return loss

class WeightDecayScheduler(fai.LearnerCallback):
    _order = -21 # Needs to run before distributed
    def __init__(self, learn:fai.Learner, model_loss_func, 
                 wd_max:float, div_factor:float=8., pct_start:float=0.3, final_div:float=50., 
                 tot_epochs:int=None, start_epoch:int=None):
        super().__init__(learn)
        self.wd_max, self.div_factor, self.pct_start, self.final_div = wd_max, div_factor, pct_start, final_div
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs
        
        if fai.is_listy(self.wd_max): self.wd_max = np.array(self.wd_max)
        
        self.model_loss_func = model_loss_func

    def steps(self, *steps_cfg:fai.StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [fai.Scheduler(step, n_iter, func=func)
                for (step,(n_iter,func)) in zip(steps_cfg, self.phases)]

    def on_train_begin(self, n_epochs:int, epoch:int, **kwargs:fai.Any)->None:
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        
        self.start_epoch = fai.ifnone(self.start_epoch, epoch)
        self.tot_epochs = fai.ifnone(self.tot_epochs, n_epochs)
        
        n = len(self.learn.data.train_dl) * self.tot_epochs
        a1 = int(n * self.pct_start)
        a2 = n-a1
        self.phases = ((a1, fai.annealing_cos), (a2, fai.annealing_cos))
        
        low_wd = self.wd_max/self.div_factor
        final_wd = self.wd_max/self.final_div
        self.wd_scheds = self.steps((low_wd, self.wd_max), (self.wd_max, final_wd))
        
        if not hasattr(self.learn, 'weight_decay_called'):
            self.learn.model = WDModelWrapper(self.learn.model, self.model_loss_func, low_wd)
            self.learn.loss_func = WDLossWrapper(self.learn.loss_func)
            self.learn.weight_decay_called = True
        
        
        self.model = self.learn.model
        self.model.mult = self.wd_scheds[0].start
        self.idx_s = 0
        
        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:fai.Any)->None:
        if train:
            if self.idx_s >= len(self.wd_scheds): return {'stop_training': True, 'stop_epoch': True}
            self.model.mult = self.wd_scheds[self.idx_s].step()
            #print(f"wd = {self.model.mult}")
            
            if self.wd_scheds[self.idx_s].is_done:
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs:fai.Any)->None:
        if epoch > self.tot_epochs: return {'stop_training': True}
