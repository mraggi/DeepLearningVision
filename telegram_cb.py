from telegram.ext import Updater
from fastai.callback.all import Callback
from fastai.learner import Recorder
from time import time
from fastprogress.fastprogress import format_time
from warnings import warn
import threading

class TelegramNotifier(Callback):
    "A `LearnerCallback` that notifies you via telegram when an epoch is done, and its associated metrics."
    run_after=Recorder
    def __init__(self, pre_msg_txt="", chat_id:int=None, token:str=None, report_every_nth_batch=0): 
        super().__init__()
        if token is None:
            raise ValueError("Please supply your bot token")
        if chat_id is None:
            raise ValueError("Please supply the chat id you wish to send the notifications to")
        self.updater = Updater(token=token)
        self.chat_id = chat_id
        self.bot = self.updater.bot 
        self.pre_msg_txt = pre_msg_txt
        
        if report_every_nth_batch > 0:
            self.after_batch = self._after_batch
            self.report_every_nth_batch = report_every_nth_batch

    def before_epoch(self):
        self.start_epoch_time = time()
        self.n_batch = 0
    
    def _snd(self, msg):
        self.bot.send_message(chat_id=self.chat_id, text=msg)
    
    def async_send_message(self, msg):
        threading.Thread(target=self._snd, args=(msg,)).start()
        
    def _after_batch(self):
        self.n_batch += 1
        batch,rep = self.n_batch,self.report_every_nth_batch
        try:
            if batch%rep == 0:
                msg = self.pre_msg_txt
                msg += f" | {batch} TL: {self.learn.smooth_loss:.4f}"
                self.async_send_message(msg)
        except Exception as e:
            warn("Could not deliver message. Error: " + str(e), RuntimeWarning)
    
    def after_epoch(self):
        try:
            rec = self.learn.recorder
            
            names = [str(m) for m in rec.metric_names]
            values = [rec.epoch] + [f"{r:.3f}" for r in rec.final_record] + [format_time(time()-self.start_epoch_time)]
            if len(names) == len(values)+1:
                names = names[:1] + names[2:] # remove train_loss when validating only
            msg = '\n'.join([f"{name}: {value}" for name,value in zip(names,values)])
            msg = self.pre_msg_txt + '\n\n' + msg + '\n-------------\n'

            self.async_send_message(msg)
        except Exception as e:
            warn("Could not deliver message. Error: " + str(e), RuntimeWarning)

