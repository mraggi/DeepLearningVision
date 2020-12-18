import torch
import torch.nn as nn
import torch.nn.functional as F
import fastai.vision.all as fv

def create_identity_conv_weight(ni,no=None,k=3,s=1,device='cpu'):
    if no is None: no=ni
    noo = no if no >= ni else ni
    T = torch.zeros(noo,ni,k,k,device=device)
    I = torch.arange(ni,device=device)
    a = (k-s)//2
    b = a+s
    T[I,I,a:b,a:b] = 1/s**2
    return T[:no]

def init_conv_weight(w,nc,g,s):
    no,ni,k,k = w.shape
    no //= g
    with torch.no_grad():
        for i in range(g):
            c = min(ni,nc,no)
            start = i*no
            end = start + c
            w[start:end] = create_identity_conv_weight(ni,c,k,s,w.device)
            nc -= c
            if nc == 0:
                break

def init_id(linear, wd=1., nc = None):
    linear.init_id = True
    linear.wd = wd

    no,ni = linear.weight.shape

    if nc is None: nc = min(no,ni)
    assert(0 < nc <= min(no,ni))
    linear.nc = nc # save for weight decay

    if linear.bias is not None:
        nn.init.constant_(linear.bias, 0.)
    with torch.no_grad():
        nn.init.constant_(linear.weight[:nc], 0.)
        linear.weight[:nc,:nc] = torch.eye(nc)

def init_id_conv(conv, wd=1., nc = None):
    conv.init_id = True
    conv.wd = wd

    no,ni,k,k = conv.weight.shape
    g = conv.groups
    s = conv.stride[0]

    if nc is None: nc = min(no,ni*g)
    assert(0 < nc <= min(no,ni*g))
    conv.nc = nc # save for weight decay

    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.)

    init_conv_weight(conv.weight,nc,g,s)
    
def init_bn_to_0(bn, wd=1.):
    nn.init.constant_(bn.weight, 0.)
    bn.init_zero = True
    bn.wd = wd

def _decay_to_0(T,pre_div=2.):
    t = T/pre_div
    return (t*t).mean()

def bn_loss(bn,surge_protection):
    if not bn.affine: return 0.
    w = bn.weight.float()
    b = bn.bias.float()
    
    wd = bn.wd if hasattr(bn,'wd') else surge_protection
    init_zero = (hasattr(bn,'init_zero') and bn.init_zero)
    pre_div = 1. if init_zero else 2.
    
    t = torch.zeros_like if init_zero else torch.ones_like
    return F.mse_loss(w,t(w))*wd + _decay_to_0(b,pre_div=pre_div)*surge_protection

def _conv_weight_decay(c,surge_protection):
    wd = c.wd if hasattr(c,'wd') else surge_protection
    init_id = (hasattr(c,'init_id') and c.init_id)
    if init_id:
        other = c.weight.clone().detach()
        nc,g,s = c.nc, c.groups, c.stride[0]
        init_conv_weight(other,nc,g,s)
        return F.mse_loss(c.weight,other)*wd
    else:
        return _decay_to_0(c.weight)*wd

def conv_loss(c,surge_protection):
    loss = _conv_weight_decay(c,surge_protection)
    
    if c.bias is not None and surge_protection > 0:
        b = c.bias.float()
        loss += surge_protection*_decay_to_0(b)

    return loss

def model_loss(model,surge_protection=0.05):
    loss = 0.
    modules = fv.flatten_model(model)
    for m in modules:
        if isinstance(m, nn.Conv2d):
            loss += conv_loss(m,surge_protection)
        elif isinstance(m, nn.BatchNorm2d):
            loss += bn_loss(m,surge_protection)
    return loss

class WeightDecaySmart(fv.Callback):
    run_after,run_valid = fv.TrainEvalCallback,False
    
    def __init__(self,pct=0.3,start=0.002,middle=0.004,end=0.0,surge_protection=0.05):
        self.sched = fv.combined_cos(pct,start,middle,end)
        self.surge_protection = surge_protection

    def after_loss(self):
        wd = self.sched(self.pct_train)
        self.learn.loss = self.learn.loss + wd*model_loss(self.learn.model,surge_protection=self.surge_protection)
        #print(f"wtf {self.learn}")
