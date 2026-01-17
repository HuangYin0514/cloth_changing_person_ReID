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
    def __init__(self, in_dim=2048, num_classes=None):
        super(Instance_Norm, self).__init__()
        self.IN = nn.InstanceNorm2d(in_dim, affine=True)
        self.mask1 = Mask(2048)

        self.alpha = nn.Parameter(torch.tensor(1.0))  # 差值项权重

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
        # in_feat_map = self.IN(feat_map)

        # diff_feat_map = feat_map - in_feat_map

        # mask = self.mask1(diff_feat_map)
        # useful_feat_map = diff_feat_map * mask
        # unuseful_feat_map = diff_feat_map * (1 - mask)

        # feat_map = in_feat_map + useful_feat_map
        # unuseful_feat_map = in_feat_map + unuseful_feat_map

        # return in_feat_map, feat_map, unuseful_feat_map

        # ---------------------------------------------
        # in_feat_map = self.IN(feat_map)

        # diff_feat_map = feat_map - in_feat_map

        # mask = self.mask1(diff_feat_map)
        # useful_feat_map = diff_feat_map * mask
        # unuseful_feat_map = diff_feat_map * (1 - mask)

        # feat_map = (in_feat_map + useful_feat_map) * 0.1 + feat_map * 0.9
        # unuseful_feat_map = (in_feat_map + unuseful_feat_map) * 0.1 + feat_map * 0.9

        # return in_feat_map, feat_map, unuseful_feat_map

        # ---------------------------------------------
        res_feat_map = feat_map
        in_feat_map = self.IN(feat_map)

        mask = self.mask1(feat_map)

        feat_map = in_feat_map + self.alpha * mask * (feat_map - in_feat_map)

        return feat_map + res_feat_map

    def loss(self, in_feat_map, feat_map, unuseful_feat_map):
        in_feat_score = F.softmax(self.classifier(self.avgpool(in_feat_map).view(unuseful_feat_map.size(0), -1)))
        feat_score = F.softmax(self.classifier(self.avgpool(feat_map).view(feat_map.size(0), -1)))
        unuse_feat_score = F.softmax(self.classifier(self.avgpool(unuseful_feat_map).view(unuseful_feat_map.size(0), -1)))

        loss = 0.01 * self.get_causality_loss(self.get_entropy(in_feat_score), self.get_entropy(feat_score), self.get_entropy(unuse_feat_score))
        return loss

    def get_entropy(self, p_softmax):
        # exploit ENTropy minimization (ENT) to help DA,
        mask = p_softmax.ge(0.000001)
        mask_out = torch.masked_select(p_softmax, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(p_softmax.size(0))

    def get_causality_loss(self, x_IN_entropy, x_useful_entropy, x_useless_entropy):
        self.ranking_loss = torch.nn.SoftMarginLoss()
        y = torch.ones_like(x_IN_entropy)
        return self.ranking_loss(x_IN_entropy - x_useful_entropy, y) + self.ranking_loss(x_useless_entropy - x_IN_entropy, y)


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
