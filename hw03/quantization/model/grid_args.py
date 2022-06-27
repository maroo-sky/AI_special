import torch
import torch.nn as nn

class grid_args():
    def __init__(self,optimizer,
                 num_epoch,
                 lr_rate,
                 weight_decay,
                 train_batch_size, **kwargs):
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.lr_rate = lr_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.ae_num_epoch = None
        self.ae_lr_rate = None
        self.ae_weight_decay = None

        if len(kwargs) != 0:
            self.ae_num_epoch = kwargs["ae_num_epoch"]
            self.ae_lr_rate = kwargs["ae_lr_rate"]
            self.ae_weight_decay = kwargs["ae_weight_decay"]