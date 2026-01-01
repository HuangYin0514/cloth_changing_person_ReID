import torch
import torch.nn as nn
import util


class Empty_Module(nn.Module):
    def __init__(self):
        super(Empty_Module, self).__init__()

    def forward(self, x):
        return
