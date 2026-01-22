import torch
from torch import nn
from torch.nn import functional as F

"""
os block https://github.com/MatthewAbugeja/osnet/blob/master/torchreid/models/osnet.py
https://arxiv.org/pdf/1910.06827v5

"""


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.se = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=True),
#             nn.ReLU(),
#             nn.Conv2d(channel // reduction, channel, 1, bias=True),
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_result = self.avgpool(x)
#         avg_out = self.se(avg_result)
#         output = self.sigmoid(avg_out)
#         return output * x


# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.sigmoid(self.conv1(out))
#         return out * x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Conv2d(2 * channel, 1 * channel, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = self.avgpool(x)
        max_result = self.maxpool(x)
        pool_result = torch.cat([avg_result, max_result], dim=1)
        avg_out = self.se(pool_result)
        output = self.sigmoid(avg_out)
        return output * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelAttention(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c)
        x3 = self.conv3(x2)
        out = x3 + identity
        return F.relu(out)


if __name__ == "__main__":
    x = torch.randn(1, 2048, 24, 12)
    model = OSBlock(in_channels=2048, out_channels=2048)
    print(model(x).shape)
