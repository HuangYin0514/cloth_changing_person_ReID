import copy

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class Propagation(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T=4):
        super(Propagation, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        p_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        loss = self.kl(p_s, p_t)  # * (self.T ** 2)
        return loss
