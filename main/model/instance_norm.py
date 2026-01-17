import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    def __init__(self, dim, r=16):
        super(Mask, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mask = self.channel_attention(x)
        return mask


class Instance_Norm(nn.Module):
    def __init__(self, in_dim=2048):
        super(Instance_Norm, self).__init__()
        self.IN = nn.InstanceNorm2d(in_dim, affine=True)
        self.mask1 = Mask(in_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_map):
        res_feat_map = feat_map
        in_feat_map = self.IN(feat_map)

        mask = self.mask1(feat_map)

        feat_map = mask * feat_map + (1 - mask) * in_feat_map

        return feat_map


if __name__ == "__main__":
    # 创建模块实例（测试 in_dim=2048 和 in_dim=512 两种情况）
    model_2048 = Instance_Norm(in_dim=2048)
    model_512 = Instance_Norm(in_dim=512)

    # 生成测试输入（batch_size=2, channel=2048/512, height=16, width=16）
    x_2048 = torch.randn(2, 2048, 24, 12)

    # 前向传播
    try:
        out_2048 = model_2048(x_2048)
        print("模块运行正常！")
        print(f"输出维度: {out_2048.shape}")
    except Exception as e:
        print(f"模块运行出错：{e}")
