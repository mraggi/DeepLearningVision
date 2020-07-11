import fastai.basics as fai
import fastai.vision as fv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from delegation import delegates

import random

relu = nn.ReLU(inplace=False)
relu_i = nn.ReLU(inplace=True)

leaky = nn.LeakyReLU(1/16, inplace=False)
leaky_i = nn.LeakyReLU(1/16, inplace=True)

celu = nn.CELU(inplace=False)
celu_i = nn.CELU(inplace=True)

default_act = nn.CELU(inplace=False)
default_act_i = nn.CELU(inplace=True)

def num_params(model):
    total=0
    for p in model.parameters():
        num=1
        for s in list(p.size()):
            num *= s
        total += num
    return total

def avg_conv_weight(ni,k,s,device='cpu'):
    T = torch.zeros(ni,ni,k,k,device=device)
    I = torch.arange(ni,device=device)
    a = (k-s)//2
    b = a+s
    T[I,I,a:b,a:b] = 1/s**2
    
    return T

def init_avg_conv(conv,wd=1.):
    conv.init_avg = True
    conv.wd = wd
    k = conv.kernel_size[0]
    s = conv.stride[0]
    g = conv.groups
    with torch.no_grad():
        if conv.bias is not None: conv.bias.zero_()
        
        T = conv.weight
        no,ni,k,k = T.shape
        
        assert(no >= ni)
        for i in range(g):
            T[i*ni:(i+1)*ni] = avg_conv_weight(ni,k,s,T.device)

def init_bn_to_0(bn, wd=1.):
    nn.init.constant_(bn.weight, 0.)
    bn.init_zero = True
    bn.wd = wd

def init_to_ignore_act(m,ni):
    if isinstance(m,nn.Conv2d):
        nn.init.constant_(m.bias[:ni],0.0)
    if isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.bias[:ni],-0.0)

def replace_module_by_other(model,module,other):
    for child_name, child in model.named_children():
        if isinstance(child, module):
            setattr(model, child_name, other)
        else:
            replace_module_by_other(child,module,other)

def identity(x):
    return x

def conv2d(ni,no,k=3,s=1,pad="same",g=1,init='none',bias=True):
    assert(k%s == 0)
    if pad=="same":
        pad = (k-1)//2
        
    conv = nn.Conv2d(ni,no,kernel_size=k,stride=s,padding=pad,groups=g,bias=bias)
    
    if bias:
        nn.init.constant_(conv.bias,0.)
    
    if not fai.is_listy(init): init = [init]
    
    if 'linear' in init:
        nn.init.kaiming_normal_(conv.weight, nonlinearity=init)
    else:
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        
    if 'identity' in init or 'avg' in init: init_avg_conv(conv)
        
    return conv

@delegates(conv2d)
def abc(ni, no, bn=True, activation=True, act_fn=default_act, bn_init_zero=False, p = 0., **kwargs):

    layers = []
    if activation:
        layers += [act_fn]

    if p > 0:
        layers += [nn.Dropout2d(p)]

    if bn:
        layers += [nn.BatchNorm2d(ni)]
        if bn_init_zero: init_bn_to_0(layers[-1])
        
        
    layers += [conv2d(ni, no, **kwargs)]

    return layers


@delegates(abc)
def abc_block(ni,no,**kwargs):
    return nn.Sequential(*abc(ni,no,**kwargs))

@delegates(conv2d)
def cab(ni, no, bn=True, activation=True, act_fn=default_act, bn_init_zero=False, p = 0., init='relu', **kwargs):
    
    ignore_act = False
    
    if init == 'avg' or init == 'identity':
        init = [init]
        ignore_act = True
        
    
    layers = [conv2d(ni, no, init=init, **kwargs)]
    
    if ignore_act:
        init_to_ignore_act(layers[-1],ni)
    
    if activation:
        layers += [act_fn]

    if p > 0:
        layers += [nn.Dropout2d(p)]

    if bn:
        layers += [nn.BatchNorm2d(no)]
        if bn_init_zero: init_bn_to_0(layers[-1])
        if ignore_act: init_to_ignore_act(layers[-1],ni)

    return layers
    

@delegates(cab)
def cab_block(ni,no,**kwargs):
    return nn.Sequential(*cab(ni,no,**kwargs))

def _get_bn_weight(A):
    if not isinstance(A,nn.Sequential):
        return 1
    if isinstance(A[-1],nn.BatchNorm2d):
        return A[-1].weight[None,:,None,None]
    return 1

def abl(ni, no, bn=True, activation=True, act_fn=default_act_i, bn_init_zero=False, p = 0.):

    layers = []
    if activation:
        layers += [act_fn]

    if p > 0:
        layers += [nn.Dropout(p)]

    if bn:
        layers += [nn.BatchNorm1d(ni)]
        
        if bn_init_zero:
            init_bn_to_0(layers[-1])  
        
    if p > 0:
        layers += [nn.Dropout(p)]
        
    layers += [nn.Linear(ni, no)]
    
    

    return layers

def lab(ni, no, bn=True, activation=True, act_fn=default_act_i, bn_init_zero=False, p = 0.):

    layers = [nn.Linear(ni, no)]
    if activation:
        layers += [act_fn]

    if bn:
        layers += [nn.BatchNorm1d(no)]
        
        if bn_init_zero:
            init_bn_to_0(layers[-1])  

    if p > 0:
        layers += [nn.Dropout(p)]

    return layers

class PositionalInfo(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        bs,c,h,w = x.shape
        H = torch.arange(0,1,1/h)
        W = torch.arange(0,1,1/w)
        H = H[None,None,:,None].expand(bs,1,h,w).to(x)
        W = W[None,None,None,:].expand(bs,1,h,w).to(x)
        return torch.cat([H,W,x],dim=1)

class SmartAdd(nn.Module):
    def __init__(self,A,B):
        super().__init__()
        self.pathA = A
        self.pathB = B
        
    def forward(self, x):
        A, B = self.pathA, self.pathB
        #return A(x) + B(x)
        a = _get_bn_weight(A)
        b = _get_bn_weight(B)
        
        divider = torch.sqrt(a*a + b*b) + 1e-7
        divider=divider.to(x)
        
        return (A(x) + B(x))/divider
    
class Concat(nn.Module):
    def __init__(self,A,B):
        super().__init__()
        self.pathA = A
        self.pathB = B
        
    def forward(self, x):
        A, B = self.pathA, self.pathB
        
        return torch.cat((A(x),B(x)),dim=1)

def ResBlock(ni, no=None, bottle=None, s=1, g=1, use_pool = False, act_fn=default_act):
    if no is None: no = ni
    if bottle is None: bottle = ni
    
    assert(no >= ni)
    
    k = 3
    pool = identity
    
    if s == 2: k = 4
        
    if use_pool or s == 2 or no > ni:
        pool = cab_block(ni,no,k=k,s=s,act_fn=act_fn,init='avg')
    
    residual = nn.Sequential(*cab(ni,bottle,k=1,act_fn=act_fn),
                             *cab(bottle,bottle,k=k,s=s,g=g,act_fn=act_fn),
                             *cab(bottle,no,act_fn=act_fn,bn_init_zero=True))
    
    return SmartAdd(pool,residual)

class Cascade(nn.Module):
    def __init__(self,nf,steps,act_fn=default_act):
        super().__init__()
        assert(nf%steps == 0)
        t = nf//steps
        
        self.t = t
        
        modules = [abc_block(t,t,act_fn=act_fn) for _ in range(steps-1)]
        self.convs = nn.ModuleList(modules)
    
    def forward(self, x):
        t, convs = self.t, self.convs
        
        p = torch.split(x,t,dim=1)
        L = [p[0]]
        prev = 0.
        for conv, z in zip(convs, p[1:]):
            prev = conv(prev+z)
            L.append(prev)
        return torch.cat(L,dim=1)

def Res2Block(ni, no=None, bottle=None, g=4, use_pool=True, act_fn=default_act, bn=True):
    if no is None: no = ni
    if bottle is None: bottle = ni
    
    pool = identity
    if use_pool or no > ni:
        pool = cab_block(ni,no,act_fn=act_fn,init='identity')
    
    BN = nn.BatchNorm2d(no)
    init_bn_to_0(BN)
    
    residual = nn.Sequential(conv2d(ni,bottle,k=1),
                             Cascade(bottle,steps=g, act_fn=act_fn),
                             *abc(bottle,no,k=3,act_fn=act_fn),
                             act_fn,
                             BN
                            )
        
    return SmartAdd(pool,residual)

class RandomResizeLayer(nn.Module):
    def __init__(self, min_size, max_size=None, stride=32, max_diff_xy=1):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
        self.max_diff_xy = max_diff_xy
    
    def forward(self, x):
        if not self.training:
            return x
        
        min_size, max_size, stride = self.min_size, self.max_size, self.stride
        
        if max_size is None: max_size = x.shape[2]
        if min_size >= max_size: min_size = max_size
            
        rx = random.choice(range(min_size,max_size+1,stride))
        ry = rx
        
        md = self.max_diff_xy
        miny = max(min_size, rx-md*stride)
        maxy = min(max_size, rx+md*stride)
        ry = random.choice(range(miny,maxy+1,stride))
        
        #print(f"Resizing to {rx},{ry}")
        'bicubic'
        modes = ['nearest', 'bilinear', 'bicubic']
        
        mode = random.choice(modes)
        
        align_corners = random.choice((False,True))
        
        if mode == 'nearest':
            align_corners = None
        
        return F.interpolate(x, size=(rx,ry), 
                             mode=mode, 
                             align_corners=align_corners)

class Normalize(nn.Module):
    def __init__(self, mean = None, std = None):
        super().__init__()
        
        if mean is None:
            mean, std = [torch.tensor(t) for t in fv.imagenet_stats]
        
        assert(len(mean.shape) == 1)
        assert(len(std.shape) == 1)
        self.mean = mean[None,:,None,None]
        self.std = std[None,:,None,None]

    def forward(self, x):
        return (x-self.mean.to(x))/self.std.to(x)

class AdaptiveAddPool(nn.Module):
    def __init__(self, max_multiplier=1.1):
        super().__init__()
        
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.mm = max_multiplier
        
        
    def forward(self, x):
        return (self.mp(x)*self.mm + self.ap(x)).squeeze()
    
class AdaptiveSubstractPool(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.ap = nn.AdaptiveAvgPool2d(1)
        
        
    def forward(self, x):
        return (self.mp(x)*2 - self.ap(x)).squeeze()

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  
        return x*torch.tanh(F.softplus(x))
    
def add_module_to_seq_model(model, module, i="last"):
    layers = list(model)
    if i == "last":
        layers.append(module)
    else:
        layers = layers[:i] + [module] + layers[i:]
    return nn.Sequential(*layers)
