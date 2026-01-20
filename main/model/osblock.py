import torch
from torch import nn


class OSBlock(nn.Module):
    def __init__(self, shape, num_scales=4):
        super().__init__()
        in_channels = shape[0]

        self.num_scales = num_scales
        mid_channels = in_channels // num_scales

        lightconv3x3 = lambda channels: nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))

        self.conv2 = nn.ModuleList([nn.Sequential(*[lightconv3x3(mid_channels) for _ in range(i + 1)]) for i in range(num_scales)])

        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)

        self.share_attn = nn.Sequential(nn.Conv1d(2, 1, 5, 1, 2), nn.Sigmoid())
        nn.init.zeros_(self.share_attn[-2].bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = 0.0
        for i in range(self.num_scales):
            x2_ = self.conv2[i](x1)
            w = self.share_attn(torch.cat([x2_.mean((2, 3)).view(x.size(0), 1, -1), x2_.amax((2, 3)).view(x.size(0), 1, -1)], dim=1)).view(x.size(0), -1, 1, 1)
            x2 = x2 + x2_ * w

        x3 = self.conv3(x2)
        return x3


if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = OSBlock((64, 32, 32))
    print(model(x).shape)
