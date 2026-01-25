# cloth_changing_person_ReID


os block https://github.com/MatthewAbugeja/osnet/blob/master/torchreid/models/osnet.py
https://arxiv.org/pdf/1910.06827v5

196 平均池化 189复现 120epoch
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = self.avgpool(x)
        avg_out = self.se(avg_result)
        output = self.sigmoid(avg_out)
        return output * x
0.1897
0.42092

197 https://github.com/GuHY777/MFENet-VIReID/blob/main/model/mfenet_no2.py
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel * 2, channel // reduction, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = self.avgpool(x)
        max_result = self.maxpool(x)
        pool_result = torch.cat([avg_result, max_result], dim=1)
        avg_out = self.se(pool_result)
        output = self.sigmoid(avg_out)
        return output * x
0.18935
0.38776

198 cbam sa
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x
0.16138
0.375

199 197的基础上改进
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
0.19106
0.39796

200
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 1, 5, 1, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.cat(
            [
                x.mean((2, 3)).view(B, 1, C),
                x.amax((2, 3)).view(B, 1, C),
            ],
            dim=1,
        )
        out = self.sigmoid(self.conv1(out)).view(B, C, 1, 1)
        return out * x
0.15653
0.35459

201
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, 1, 0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = self.avgpool(x)
        max_result = self.maxpool(x)
        pool_result = torch.cat([avg_result, max_result], dim=1)
        avg_out = self.se(pool_result)
        output = self.sigmoid(avg_out)
        return output * x
0.18494
0.3801


202 197改进
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = self.avgpool(x)
        max_result = self.maxpool(x)
        pool_result = torch.cat([avg_result, max_result], dim=1)
        avg_out = self.se(pool_result)
        output = self.sigmoid(avg_out)
        return output * x
0.14939
0.41071


203
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Sequential(
            # 先降维：channel*2 -> channel//reduction
            nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            # 再升维：channel//reduction -> channel
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = self.avgpool(x)
        max_result = self.maxpool(x)
        pool_result = torch.cat([avg_result, max_result], dim=1)
        avg_out = self.se(pool_result)
        output = self.sigmoid(avg_out)
        return output * x
0.16194
0.37755

206 prcc数据集
0.53237
0.5357

208 25ep 开启辅助训练
0.50618
0.47954

209 去除修正模块
0.50412
0.51454

210 distmat = get_distmat(qf, gf, dist="euclidean")
euclidean cosine性能完全一致

211 Baseline r50ibn
0.53405
0.51764

212 Baseline r50
0.56711
0.55913

213 r50 完整模块
0.54958
0.53203

214 去除多尺度模块
0.56636
0.5755

215 去除多尺度+池化mean
0.50546
0.49591

[xxx 蒸馏损失*10 / 服装平滑ce]
216 213对比 完整模块 r50 性能没有变化
0.54958
0.53203

218 修复错误测试问题
