from torch.distributions import Beta
import random
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

import fastai.vision.all as fv
from fastai.vision.all import L
from Layers import RandomSizeCropAndResizeBatch


class CutMixUp(fv.Callback):
    run_after,run_valid = [RandomSizeCropAndResizeBatch,fv.RandomResizedCrop,fv.RandomResizedCropGPU],False
    def __init__(self, α_cutmix=1., α_mixup=0.4, weights=None, cutmix_random_paste=True):
        α_cutmix, α_mixup = (torch.tensor(α).float() for α in (α_cutmix,α_mixup))
        self.distrib_cutmix = Beta(α_cutmix, α_cutmix) # this is dumb if α=1
        self.distrib_mixup  = Beta(α_mixup, α_mixup)
        self.weights = weights
        self.cutmix_random_paste = cutmix_random_paste
        
    def before_fit(self):
        self.mix_y = not getattr(self.learn.loss_func, 'y_int', False)
        if not self.mix_y: 
            self.old_lf =  self.learn.loss_func
            self.learn.loss_func = self.lf

    def after_fit(self):
        if not self.mix_y: 
            self.learn.loss_func = self.old_lf

    def before_batch(self):
        x,y = self.learn.x, self.learn.y
        self.mode = random.choices(['cutmix','mixup'], weights=self.weights)[0]
        
        shuffle = torch.randperm(y.shape[0], device=y.device)
        self.x_shuffled, self.y_shuffled = x[shuffle], y[shuffle]
        
        #print("Chose mode: ", self.mode)
        if self.mode == 'cutmix': self.mix_with_shuffled_cutmix()
        elif self.mode == 'mixup': self.mix_with_shuffled_mixup()
    
        if self.mix_y:
            self.learn.yb = (torch.lerp(self.y_shuffled, y, self.λ),)

    def mix_with_shuffled_mixup(self):
        x,y = self.learn.x, self.learn.y
        λ = self.distrib_mixup.sample((x.shape[0],))
        λ = torch.max(λ,1-λ).to(x.device)
        λx = fv.unsqueeze(λ, n=len(x.shape)-1)
        self.learn.xb = (torch.lerp(self.x_shuffled, x, λx),)
        
        self.λ = fv.unsqueeze(λ, n=len(y.shape)-1)
    
    def mix_with_shuffled_cutmix(self):
        λ = self.distrib_cutmix.sample((1,)).item()
        λ = max(λ,1-λ)

        H, W = self.learn.x.shape[2:]
        
        pct = sqrt(1 - λ)
        len_h,len_w = round(H*pct), round(W*pct)

        h0,h1,w0,w1 = *self.rand_interval(0,H,len_h), *self.rand_interval(0,W,len_w)
        if self.cutmix_random_paste:
            H0,H1,W0,W1 = *self.rand_interval(0,H,len_h), *self.rand_interval(0,W,len_w)
        else:  
            H0,H1,W0,W1 = h0,h1,w0,w1
            
        self.learn.xb[0][:, :, h0:h1, w0:w1] = self.x_shuffled[:, :, H0:H1, W0:W1]
        
        boxArea = (h1-h0)*(w1-w0)
        self.λ = (1 - boxArea/(W*H)) # 'cause of rounding errors. Maybe useless.
    
    def lf(self, pred, *yb):
        if not self.training: 
            return self.old_lf(pred, *yb)
        with fv.NoneReduce(self.old_lf) as old_lf:
            loss = torch.lerp(old_lf(pred, self.y_shuffled), old_lf(pred,*yb), self.λ)
        return fv.reduce_loss(loss, getattr(self.old_lf, 'reduction', 'mean'))
    
    def rand_interval(self,a,b,length):
        l = random.randrange(a, b-length)
        r = l + length
        return l, r
