import fastai.basics as fai
import fastai.vision as fv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import gc
import pandas as pd
from fastai.script import *
from fastai.distributed import *
from fastai.callbacks import *

from callbacks import *
from weightdecay import *
from Layers import *
from xranger import *
from model import *
from data import *

from functools import partial

from fastai.distributed import *
import argparse
import time
import warnings

warnings.simplefilter("ignore")

def train(model,model_name,src,sz,bs,epochs,lr,div_factor,final_div,pct_start,mixup,fp16,finetune,wd,start_epoch,local_rank,freeze):
    data = load_data(src, img_size=sz, batch_size=bs//2,finetune=finetune)

    callback_fns = []
    
    if local_rank == 0:
        save_cb = partial(SaveOnlyModelAfterEpoch,name=model_name, model=model)
        callback_fns.append(save_cb)
    
    if not finetune:
        resize_cb = partial(ResizeCallback,min_size=sz//2)
        callback_fns += [resize_cb]

    if wd > 0:
        wdsched_cb = partial(WeightDecayScheduler,
                                model_loss_func=model_loss,
                                wd_max=wd,
                                div_factor=10,final_div=100,pct_start=0.75)
            
        callback_fns += [wdsched_cb]

    layer_groups = [nn.ModuleList(fv.flatten_model(model[i])) for i in range(len(model))]

    optimizer = partial(XRanger,betas=(0.95,0.99))
    
    learn = fv.Learner(data, model, 
                        wd = 0, true_wd = False, bn_wd = False, #I'm doing my own weight decay!
                        opt_func = optimizer,
                        metrics=[fv.top_k_accuracy,fv.accuracy],
                        layer_groups = layer_groups,
                        loss_func = LabelSmoothingCrossEntropyFixed(),
                        callback_fns = callback_fns)
    
    if fp16:
        learn.to_fp16()
        print_once("Doing fp16!")
    else:
        learn.to_fp32()
        
    if mixup:
        learn.mixup()
        print_once("Doing mixup!")
    
    
    if freeze:
        print("Training only last layer!")
        learn.freeze()
        
    learn.to_distributed(local_rank);
    
    gc.collect()
    torch.cuda.empty_cache()
    
    learn.fit_one_cycle(epochs, slice(lr//4,lr), 
                        div_factor=div_factor, final_div=final_div, pct_start=pct_start, 
                        start_epoch=start_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int)
    
    parser.add_argument("--data_path", type=str)
    
    parser.add_argument("--model_name", type=str,default="sv")
    parser.add_argument("--model", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    parser.add_argument("--phases",type=str,default="phases.csv")
    parser.add_argument("--num_epochs_frozen",type=int,default=2,help="If loading a model and changing num categories, train for this many epochs first.")
    parser.add_argument("--start_phase",type=int,default=0)
    parser.add_argument("--start_epoch",type=int,default=0)

    args = parser.parse_args()

    local_rank = args.local_rank

    save_path = args.save_path

    if save_path == "":
        save_path = f"{args.data_path}/models"

    os.makedirs(save_path,exist_ok=True)

    path = Path(args.data_path)
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    src = load_src(path)
    model, should_train_last_layer = get_model(src.c, args.model_name, args.model)
    model = model.cuda();
    
    def print_once(t):
        if local_rank == 0:
            print(t)

    print_once(f"Loaded model with {num_params(model)} parameters")

    phases = pd.read_csv(args.phases, skipinitialspace=True)
    phases.columns = phases.columns.str.strip()
    
    if should_train_last_layer:
        print_once(f"Since we loaded a model and then change num_cats, we will train the last layer for {args.num_epochs_frozen} epochs")
        train(model,model_name=f"{save_path}/00_pretrain",src=src,
                    sz=224,bs=512,epochs=args.num_epochs_frozen,
                    lr=1e-2,div_factor=1,final_div=1000,pct_start=0.5,
                    mixup=False,fp16=True,finetune=False,
                    wd=0.1,start_epoch=0,local_rank=local_rank,
                    freeze=True)

    first_time = True
    for row in phases.iterrows():
        phase = row[0]
        if phase < args.start_phase:
            continue
        sz,bs,epochs,lr,div_factor,final_div,pct_start,mixup,fp16,finetune,wd=row[1]
        
        sz,bs,epochs,mixup,fp16,finetune = [int(x) for x in [sz,bs,epochs,mixup,fp16,finetune]]
        
        print_once(f"\n\n******* STARTING PHASE {phase} with size {sz} for {epochs} epochs! **********\nParameters:")
        
        for var_name,val in zip(phases.columns,[sz,bs,epochs,lr,div_factor,final_div,pct_start,mixup,fp16,finetune,wd]):
            print_once(f"\t{var_name} = {val}")
        print_once("\n\n")
        
        start_epoch = args.start_epoch if first_time else 0
        first_time = False
        
        model_name = f"{save_path}/{phase}_model_{sz}_e"
        
        train(model,model_name=model_name,
              sz=sz,bs=bs,epochs=epochs,src=src,
              lr=lr,div_factor=div_factor,final_div=final_div,pct_start=pct_start,
              mixup=mixup,fp16=fp16,finetune=finetune,wd=wd,start_epoch=start_epoch,
              local_rank=local_rank,
              freeze=False)
