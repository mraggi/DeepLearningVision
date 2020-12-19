import fastai.vision.all as fv
import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path
from Layers import *

def Pequeña(c_out, filters=[32, 64, 128, 256, 384, 1024, 640]):
    f0, f1, f2, f3, f4, f5, f6 = filters
    initial = nn.Sequential(Normalize(),
                            conv2d(3, f0, k=4, s=2, init='avg'),
                            ResBlock(f0, f1, bottle=f0, s=2),
                            ResBlock(f1)
                           )

    blockA = nn.Sequential(ResBlock(f1, f2, s=2), 
                           ResBlock(f2,g=4),
                           ResBlock(f2,g=2),
                           ResBlock(f2),
                           )
    
    blockB = nn.Sequential(ResBlock(f2, f3, s=2),
                           Res2Block(f3),
                           ResBlock(f3, g=4),
                           ResBlock(f3, g=2),
                           ResBlock(f3,bottle=f2),
                           fv.PooledSelfAttention2d(f3)
                           )
    
    blockC = nn.Sequential(ResBlock(f3, f4, s=2),
                           ResBlock(f4, g=4),
                           ResBlock(f4, g=2),
                           ResBlock(f4,bottle=f2),
                           ResStairBlock(f4,groups='full'),
                           ResBlock(f4,g=4),
                           ResBlock(f4,g=2),
                           ResBlock(f4,bottle=f2+f1),
                           *abc(f4, f5, k=1, s=1, pad=0),
                           leaky,
                           nn.BatchNorm2d(f5)
                           )
    
    classifier = nn.Sequential(SmartPool(),
                               *lab(f5,f6),
                               nn.Linear(f6,c_out)
                               )
    
    return nn.Sequential(initial, blockA, blockB, blockC, classifier) 

def pequeña_splitter(m):
    return [list(p.parameters()) for p in m]

def load_pequeña(cats, file=None, **kwargs):
    if file is None:
        model = Pequeña(cats,**kwargs)
    else:
        state_dict = torch.load(file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        
        key = list(state_dict.keys())[-1]
        out = state_dict[key].shape[0]
        model = Pequeña(out, **kwargs)
        model.load_state_dict(state_dict)
        if out != cats:
            print("# categories change, so replacing last layer")
            last_nf = model[-1][-1].in_features
            model[-1][-1] = nn.Linear(last_nf, cats)
        else:
            print("# categories stayed the same.")
    print(f"Correctly loaded model PEQUEÑA with {num_params(model)/10**6:.3f}M parameters")
    return model
