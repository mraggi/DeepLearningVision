import torch
import torch.nn as nn
from fastai.torch_basics import flatten_model
from warnings import warn

def incrust_conv(A:nn.Conv2d, B:nn.Conv2d):
    Ao, Ai, Akh, Akw = A.weight.shape
    Bo, Bi, Bkh, Bkw = B.weight.shape
    mi,mo = min(Ai,Bi),min(Ao, Bo)
    Mi,Mo = max(Ai,Bi),max(Ao, Bo)
    
    assert(Akh <= Bkh and Akw <= Bkw)
    assert((A.groups == B.groups == 1) or (Ao == Bo and Ai == Bi))
    
    with torch.no_grad():
        if (A.weight.shape == B.weight.shape):
            B.weight.copy_(A.weight)
            if B.bias is not None:
                B.bias.copy_(A.bias)
            if hasattr(A,'weight_u'):
                B.weight_u.copy_(A.weight_u)
            if hasattr(A,'weight_v'):
                B.weight_v.copy_(A.weight_v)
            if hasattr(A,'weight_orig'):
                B.weight_orig.copy_(A.weight_orig)
            return
        
        skh = (Bkh - Akh)//2
        skw = (Bkw - Akw)//2
        sKh = skh + Akh
        sKw = skw + Akw
        
        B.weight[:mo,:mi].zero_()
        B.weight[:mo,:mi,skh:sKh,skw:sKw] = A.weight[:mo,:mi]
        
        if Ai < Bi:
            B.weight[:mo,mi:].zero_()
        
        if B.bias is not None:
            bias = A.bias if A.bias is not None else torch.zeros(Ao)
            B.bias[:mo] = bias[:mo]
        
        
        if Bi < Ai:
            print(f"Can't really incrust because Bi = {Bi} < {Ai} = Ai")
        if Bo < Ao:
            print(f"Dropping {Ao-Bo} filters from output because Bo = {Bo} < {Ao} = Ao")

def incrust_lin(A: nn.Linear, B: nn.Linear):
    Ao, Ai = A.weight.shape
    Bo, Bi = B.weight.shape
    mi,mo = min(Ai,Bi),min(Ao, Bo)
    Mi,Mo = max(Ai,Bi),max(Ao, Bo)
    with torch.no_grad():
        B.weight[:mo,:mi] = A.weight[:mo,:mi]
        if Ai < Bi: 
            B.weight[:mo,mi:].zero_()
            
        if B.bias is not None:
            bias = A.bias if A.bias is not None else torch.zeros_like(B.bias)
            B.bias[:mo] = bias[:mo]
            
        if Bi < Ai:
            print(f"Can't really incrust because Bi = {Bi} < {Ai} = Ai")
        if Bo < Ao:
            print(f"Dropping {Ao-Bo} filters from output because Bo = {Bo} < {Ao} = Ao")


def incrust_bn(A, B):
    Ao = A.weight.shape[0]
    Bo = B.weight.shape[0]
    mo = min(Ao, Bo)
    with torch.no_grad():
        B.weight[:mo] = A.weight[:mo]
        B.bias[:mo] = A.bias[:mo]
        B.running_mean[:mo] = A.running_mean[:mo]
        B.running_var[:mo] = A.running_var[:mo]
        B.num_batches_tracked = A.num_batches_tracked

        if Bo < Ao:
            print(f"Dropping {Ao-Bo} filters from output because Bo = {Bo} < {Ao} = Ao")

def incrust_layer(A, B):
    t = type(A)
    assert(t == type(B))
    if t == nn.Conv2d:
        incrust_conv(A,B)
    elif t == nn.Linear:
        incrust_lin(A,B)
    elif t == nn.BatchNorm2d or t == nn.BatchNorm1d:
        incrust_bn(A,B)
        
        #print(f"Not incrusting {t}")

def incrust(A:nn.Module,B:nn.Module):
    with torch.no_grad():
        for p,q in zip(A.parameters(),B.parameters()):
            if p.shape == q.shape:
                q.copy_(p)
    fA = flatten_model(A)
    fB = flatten_model(B)
    for a,b in zip(fA,fB):
        incrust_layer(a,b)
