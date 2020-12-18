import tensorboard
from torch.utils.tensorboard import SummaryWriter
from fastai.callback.all import Callback
from fastai.learner import Recorder
from fastai.torch_core import rank_distrib
from fastai.basics import ifnone
from fastai.basics import listify
import torch

class SimpleTensorBoardCallback(Callback):
    run_after=Recorder
    def __init__(self, log_dir=None, report_every_nth=8, scale=128, train_metrics = None, train_metric_names = None):
        super().__init__()
        self.log_dir = log_dir
        self.report_every_nth = report_every_nth
        self.scale = scale
        
        self.train_metrics = train_metrics
        self.train_metric_names = train_metric_names
        
    def _normalized_batch_n(self):
        return self.learn.train_iter*self.learn.dls.train.bs//self.scale
    
    def _smooth(self, name, v):
        λ = 0.9
        prevSum, prevCount = self.smooth_dict[name]
        weightedSum = v + λ*prevSum
        weightedCount = 1 + λ*prevCount
        self.smooth_dict[name] = (weightedSum,weightedCount)
        return weightedSum/weightedCount
    
    def get_name(self, s):
        if hasattr(s,'__name__'):
            return s.__name__
        else:
            return str(s)
    
    def before_fit(self):
        self.run = (rank_distrib()==0) and not (hasattr(self.learn, 'lr_finder') or hasattr(self, "gather_preds"))

        if self.run:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.train_metrics = listify(ifnone(self.train_metrics,self.learn.loss_func))
            self.train_metric_names = listify(ifnone(self.train_metric_names,[self.get_name(s) for s in self.train_metrics]))
            self.train_metric_names = ['train_'+name for name in self.train_metric_names]
            
            self.smooth_dict = {name:(0,0) for name in self.train_metric_names}
    
    def after_batch(self):
        if self.training and (self.learn.train_iter % self.report_every_nth == 0):
            t = self._normalized_batch_n()
            #print("metric names = ", self.train_metric_names)
            #print("metrics = ", self.train_metrics)
            with torch.no_grad():
                if hasattr(self.learn,'smooth_loss'):
                    self.writer.add_scalar('smooth_loss', self.learn.smooth_loss.item(), t)

                for name,metric in zip(self.train_metric_names, self.train_metrics):
                    #print(f"adding {name}")
                    pred,yb = self.learn.pred, self.learn.yb
                    value = metric(pred,*yb).item()
                    self.writer.add_scalar(name, self._smooth(name,value), t)
                    #print(f"added {name}")

        #for i,h in enumerate(self.opt.hypers):
        #    for k,v in h.items(): self.writer.add_scalar(f'{k}_{i}', v, t)
        
    def after_epoch(self):
        try:
            t = self._normalized_batch_n()
            names = self.recorder.metric_names[2:-1]
            log = self.recorder.log[2:-1]
            if len(names) != len(log):
                log = self.recorder.log[1:-1]
            if len(names) != len(log):
                print(f"names = {names} and log = {log}")
                log = self.recorder.log[1:-1]
                return
            for name,value in zip(names,log):
                self.writer.add_scalar(name, value, t)
        except:
            pass
    
    def after_fit(self):
        if self.run: 
            self.writer.close()
