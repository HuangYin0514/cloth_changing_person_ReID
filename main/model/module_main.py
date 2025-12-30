import torch
import torch.nn as nn
import util


class Empty_Module(nn.Module):
    def __init__(self):
        super(Empty_Module, self).__init__()

    def forward(self, x):
        return


class Part_Block(nn.Module):
    def __init__(self):
        super(Part_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=2048, out_channels=4, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x
