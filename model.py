import torch
import torch.nn as nn
import fastai.vision as fv
from Layers import *

def rgb2hsv(rgb, eps = 1e-7):    
    hsv = torch.zeros_like(rgb)
    
    R,G,B = rgb[:,0],rgb[:,1],rgb[:,2]
    
    MAX, iM = rgb.max(1)
    MIN, im = rgb.min(1)
    D = MAX - MIN + eps
    
    isR, isG, isB = (iM==0).float(), (iM==1).float(), (iM==2).float()

    hsv[:,2] = MAX # value
    hsv[:,1] = D/(MAX+eps) # saturation
    
    hsv[:,0] = (((G-B)/D)*isR + (2. + (B-R)/D)*isG +  (4. + (R-G)/D)*isB)/6.
    hsv[:,0][hsv[:,0] < 0.] += 1.

    return hsv

class ExtractSV(nn.Module):
    def forward(self, x):
        hsv = rgb2hsv(x)
        return x[:,1:]

def sv_model(c_out, filters=[24,48,96,128,256,512], act_fn=relu):
    f0, f1, f2, f3, f4, f5 = filters
    initial = nn.Sequential(ExtractSV(),
                            nn.BatchNorm2d(2),
                            conv2d(2, f0, k=6, s=2, init='avg'), 
                            )

    blockA = nn.Sequential(nn.MaxPool2d(2), 
                           nn.ReLU(inplace=True), 
                           nn.BatchNorm2d(f0),
                           *cab(f0, f1, init='identity', act_fn=act_fn),
                           )

    blockB = nn.Sequential(ResBlock(f1, f2, s=2, act_fn=act_fn), 
                           cab_block(f2,f2, act_fn=act_fn,init='identity'),
                           ResBlock(f2, g=8, act_fn=act_fn),
                           cab_block(f2,f2, act_fn=act_fn,init='identity'),
                           ResBlock(f2, g=2, act_fn=act_fn)
                           )
    
    blockC = nn.Sequential(PositionalInfo(),
                           ResBlock(f2+2, f3, s=2, act_fn=act_fn), 
                           cab_block(f3, f3, g=2, act_fn=act_fn,init='identity'),
                           ResBlock(f3, bottle=f3//2, g=4, act_fn=act_fn),
                           Res2Block(f3, bottle=f3//2, act_fn=act_fn),
                           ResBlock(f3, bottle=f3//2, g=2, act_fn=act_fn)
                           )
    
    blockD = nn.Sequential(ResBlock(f3, f4, s=2, act_fn=act_fn),
                           Res2Block(f4,f4, g=2, act_fn=act_fn),
                           ResBlock(f4, bottle=f3, g=4, act_fn=act_fn),
                           Res2Block(f4, act_fn=act_fn)
                           )
    
    classifier = nn.Sequential(ResBlock(f4, f5, bottle=f4, g=4, use_pool=True, act_fn=act_fn),
                               ResBlock(f5, bottle=f5//2, g=2, act_fn=act_fn),
                               AdaptiveSubstractPool(),
                               *abl(f5, c_out, activation=False)
                               )
    
    return nn.Sequential(initial, blockA, blockB, blockC, blockD, classifier)

def vgg_sv(c_out):
    initial = nn.Sequential(ExtractSV(),
                            nn.BatchNorm2d(2),
                            cab_block(2, 64, act_fn=relu_i), 
                            cab_block(64, 64, act_fn=relu_i), 
                            )

    blockA = nn.Sequential(nn.MaxPool2d(2), 
                           cab_block(64, 128, act_fn=relu_i),
                           cab_block(128, 128, act_fn=relu_i),
                           )

    blockB = nn.Sequential(nn.MaxPool2d(2),
                           cab_block(128, 256, act_fn=relu_i),
                           cab_block(256, 256, act_fn=relu_i),
                           cab_block(256, 256, act_fn=relu_i),
                           )
    
    blockC = nn.Sequential(nn.MaxPool2d(2),
                           cab_block(256, 512, act_fn=relu_i),
                           cab_block(512, 512, act_fn=relu_i),
                           cab_block(512, 512, act_fn=relu_i),
                           )
    
    blockD = nn.Sequential(nn.MaxPool2d(2),
                           cab_block(512, 512, act_fn=relu_i),
                           cab_block(512, 512, act_fn=relu_i),
                           cab_block(512, 512, act_fn=relu_i),
                           nn.MaxPool2d(2)
                           )
    
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                               fv.Flatten(),
                               *abl(512, c_out, activation=False)
                               )
    
    return nn.Sequential(initial, blockA, blockB, blockC, blockD, classifier)

def _create_model(num_classes, name,**kwargs):
    if name == 'sv':
        return sv_model(num_classes, **kwargs)
    elif name == 'vgg':
        return vgg_sv(num_classes,**kwargs)
    else:
        raise 'Not implemented'

def _get_num_classes(model_dict):
    return model_dict[list(model_dict)[-1]].shape[0]

def _load_model(num_classes, name, load_from, **kwargs):
    model_dict = torch.load(load_from)
    c = _get_num_classes(model_dict)
    last_layer_new = (c != num_classes)
    
    model = _create_model(c, name, **kwargs)
    model.load_state_dict(model_dict)
    
    if last_layer_new:
        print(f"WARNING: Loaded model has a different number of classes ({c}) than data ({num_classes}). Changing last layer to match...")
        last_layer = model[-1][-1]
        n_in = last_layer.in_features
        model[-1][-1] = nn.Linear(n_in, num_classes)
    else:
        print(f"Loaded model with SAME number of classes ({c}).")
    should_train_last_layer = last_layer_new
    return model, should_train_last_layer

def get_model(num_classes, name='sv', load_from='', **kwargs):
    if load_from != '':
        return _load_model(num_classes, name=name, load_from=load_from)
    else:
        return _create_model(num_classes, name, **kwargs), False # False means "should not train last layer separately"
    
    

