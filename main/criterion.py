import torch.nn as nn
from loss.center_triplet_loss import CenterTripletLoss
from loss.hcc import hcc
from loss.ori_triplet_loss import OriTripletLoss


class Criterion:
    def __init__(self, config):
        self.name = "Criterion"
        self.load_criterion(config)

    def load_criterion(self, config):
        self.id = nn.CrossEntropyLoss()
        self.tri = OriTripletLoss(batch_size=config.DATALOADER.BATCHSIZE, margin=0.3)
        self.hcc = hcc(margin_euc=0.6, margin_kl=6)
        self.ctl = CenterTripletLoss(batch_size=config.DATALOADER.BATCHSIZE, margin=0.3)
