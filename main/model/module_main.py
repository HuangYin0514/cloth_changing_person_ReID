import torch
import torch.nn as nn
import util


class Empty_Module(nn.Module):
    def __init__(self):
        super(Empty_Module, self).__init__()

    def forward(self, x):
        return


class MaxAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)

        return torch.cat((max_f, avg_f), 1)
