import fastai.basics as fai
import fastai.vision as fv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import gc

from fastai.script import *
from fastai.callbacks import *

from Layers import RandomResizeLayer

from functools import partial

import time

class SleepAfterEpoch(fv.LearnerCallback):
    def __init__(self, learn:fv.Learner, seconds=60):
        "Sleep so as to not overheat the GPU."
        super().__init__(learn)
        self.learn = learn
        self.seconds = seconds
    def on_epoch_end(self, epoch, **kwargs)->None:
        time.sleep(self.seconds)
        
class SleepAfterBatch(fv.LearnerCallback):
    def __init__(self, learn:fv.Learner, seconds=1):
        "Sleep so as to not overheat the GPU."
        super().__init__(learn)
        self.learn = learn
        self.seconds = seconds
    def on_batch_end(self, **kwargs)->None:
        time.sleep(self.seconds)

class SaveOnlyModelAfterEpoch(fv.LearnerCallback):
    def __init__(self, learn:fv.Learner, name, model=None, metric=None):
        "Save models on epoch end."
        super().__init__(learn)
        self.learn = learn
        self.name = name
        self.model = model if model is not None else learn.model
        self.metric = metric
    
    def get_metric(self):
        if not hasattr(self.learn,'recorder'): return "norecorder"
        
        recorder = self.learn.recorder
        
        if self.metric is None: # use validation loss
            if len(recorder.val_losses) > 0:
                val_loss = recorder.val_losses[-1].item()
                return f"{val_loss:.3f}"
            else:
                return "noloss"
        
        if not hasattr(self.learn,'metrics'): return "nometrics"
        
        metric_name = self.metric
        
        try:
            i = recorder.metrics_names.index(metric_name)
        
            return f"{recorder.metrics[-1][i].item():.3f}"
        except:
            print(recorder.metrics)
            print(metric_name)
            print(recorder.metrics_names)
            return "metricnotfound"
    
    def on_epoch_end(self, epoch, **kwargs)->None:
        acc = self.get_metric()
        name = f"{self.name}_{epoch}_{acc}.pth"
        torch.save(self.model.state_dict(), name)

class ResizeCallback(fai.LearnerCallback):
    _order = -22 # Needs to run before distributed
    def __init__(self, learn:fai.Learner, min_size, max_size=None, stride=32, max_diff_xy=1):
        super().__init__(learn)
        self.min_size,self.max_size,self.stride,self.max_diff_xy = min_size,max_size,stride,max_diff_xy

    def on_train_begin(self, **kwargs:fai.Any)->None:
        if not hasattr(self.learn, 'resize_called'):
            self.learn.model = nn.Sequential(RandomResizeLayer(self.min_size,self.max_size,self.stride,self.max_diff_xy), self.learn.model)
            self.learn.resize_called = True
    
    #def on_train_end(self, **kwargs:fai.Any)->None:
        #self.learn.model = self.learn.model[1]
