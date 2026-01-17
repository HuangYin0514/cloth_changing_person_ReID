import torch
import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class SFE(nn.Module):
    expansion = 4  # 启用 expansion，适配 ResNet 瓶颈层设计

    def __init__(self, inplanes, planes, stride=1):
        super(SFE, self).__init__()
        # 修正：将 planes 转为 bottleneck 内的维度（符合 expansion 设计）
        bottleneck_planes = planes // self.expansion

        # 1x1 降维卷积（压缩通道数）
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)

        # 4个不同感受野的分支（统一使用 bottleneck_planes 通道）
        self.conv2_1 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_3 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.conv2_4 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=1, dilation=3, padding=3, bias=False)

        self.bn2_1 = nn.BatchNorm2d(bottleneck_planes)
        self.bn2_2 = nn.BatchNorm2d(bottleneck_planes)
        self.bn2_3 = nn.BatchNorm2d(bottleneck_planes)
        self.bn2_4 = nn.BatchNorm2d(bottleneck_planes)

        # 1x1 升维卷积（恢复到 planes 通道）
        self.conv3 = nn.Conv2d(bottleneck_planes * 4, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        # 残差分支：处理维度/步长不匹配问题
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        # 初始化权重
        self.apply(weights_init_kaiming)  # 简化：对所有层批量初始化

    def forward(self, x):
        residual = x

        # 降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 4个分支并行计算
        out1 = self.relu(self.bn2_1(self.conv2_1(out)))
        out2 = self.relu(self.bn2_2(self.conv2_2(out)))
        out3 = self.relu(self.bn2_3(self.conv2_3(out)))
        out4 = self.relu(self.bn2_4(self.conv2_4(out)))

        # 拼接4个分支（通道数变为 bottleneck_planes*4）
        out = torch.cat((out1, out2, out3, out4), 1)

        # 升维
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差分支：适配维度/步长
        if self.downsample is not None:
            residual = self.downsample(residual)

        # 残差连接
        out += residual
        out = self.relu(out)

        return out


# 测试代码：验证维度是否匹配
if __name__ == "__main__":
    # 实例化模块：inplanes=64, planes=256（符合 expansion=4 的设计：64*4=256）
    model = SFE(inplanes=2048, planes=2048)
    # 输入：batch_size=2, channels=64, height=32, width=32
    x = torch.randn(2, 2048, 32, 32)
    # 前向传播
    output = model(x)
    print(f"输入维度: {x.shape}")
    print(f"输出维度: {output.shape}")  # 预期输出：torch.Size([2, 256, 32, 32])
