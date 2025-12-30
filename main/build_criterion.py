import torch
import torch.nn as nn
from loss import CrossEntropyLabelSmooth, MatchMask, TripletLoss


class Build_Criterion:
    def __init__(self, config, *args, **kwargs):
        self.build(config, *args, **kwargs)

    def build(self, config, num_classes):
        self.ce = nn.CrossEntropyLoss()
        self.ce_ls = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=torch.cuda.is_available())
        self.tri = TripletLoss(margin=0.3)
        self.match_mask = MatchMask()
