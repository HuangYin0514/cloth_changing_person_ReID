import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionCalibrator(nn.Module):
    """空间注意力校准模块
    根据全局特征图生成空间注意力权重，对待校准特征图进行空间维度校准
    """

    def __init__(self, in_channels):
        super(SpatialAttentionCalibrator, self).__init__()
        # 1x1卷积压缩通道维度，生成空间注意力的基础特征
        self.conv_global = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv_target = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

        # 空间注意力融合（使用3x3卷积捕捉空间邻域关系）
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        # 可学习的校准参数（初始化为0，避免初始时过度校准）
        self.alpha = nn.Parameter(torch.zeros(1))

        # 激活函数和归一化
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_feat, target_feat):
        """
        空间注意力校准前向传播
        Args:
            global_feat: 全局特征图, shape [B, C, H, W]
            target_feat: 待校准特征图, shape [B, C, H, W]
        Returns:
            spatial_calibrated: 空间校准后的特征图, shape [B, C, H, W]
        """
        # 1. 生成全局和目标特征的空间注意力基础图（压缩通道到1维）
        global_spatial = self.conv_global(global_feat)  # [B, 1, H, W]
        target_spatial = self.conv_target(target_feat)  # [B, 1, H, W]

        # 2. 融合生成空间注意力权重
        concat_spatial = torch.cat([global_spatial, target_spatial], dim=1)  # [B, 2, H, W]
        spatial_attention = self.attention_conv(concat_spatial)  # [B, 1, H, W]
        spatial_attention = self.sigmoid(spatial_attention)  # 归一化到0-1

        # 3. 空间校准：注意力权重 * 待校准特征 * 可学习参数 + 原特征（残差连接）
        spatial_calibrated = target_feat * spatial_attention * self.alpha + target_feat

        return spatial_calibrated


class ChannelAttentionCalibrator(nn.Module):
    """通道注意力校准模块
    根据全局特征图生成通道注意力权重，对待校准特征图进行通道维度校准
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionCalibrator, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # 全局平均池化（压缩空间维度，保留通道信息）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道注意力生成（MLP结构：压缩->激活->恢复）
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction_ratio), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction_ratio, in_channels))

        # 可学习的校准参数（初始化为0）
        self.beta = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, global_feat, target_feat):
        """
        通道注意力校准前向传播
        Args:
            global_feat: 全局特征图, shape [B, C, H, W]
            target_feat: 待校准特征图, shape [B, C, H, W]
        Returns:
            channel_calibrated: 通道校准后的特征图, shape [B, C, H, W]
        """
        # 1. 对全局和目标特征进行全局平均池化（B, C, 1, 1）
        global_pool = self.avg_pool(global_feat).view(-1, self.in_channels)  # [B, C]
        target_pool = self.avg_pool(target_feat).view(-1, self.in_channels)  # [B, C]

        # 2. 融合生成通道注意力权重
        fusion_pool = global_pool + target_pool  # 融合全局和目标的通道信息 [B, C]
        channel_attention = self.fc(fusion_pool)  # [B, C]
        channel_attention = self.sigmoid(channel_attention).view(-1, self.in_channels, 1, 1)  # [B, C, 1, 1]

        # 3. 通道校准：注意力权重 * 待校准特征 * 可学习参数 + 原特征（残差连接）
        channel_calibrated = target_feat * channel_attention * self.beta + target_feat

        return channel_calibrated


class AttentionCorrectionNetwork(nn.Module):
    """注意力修正网络（主模块）
    整合空间注意力校准和通道注意力校准，输出最终校准特征图
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionCorrectionNetwork, self).__init__()
        # 初始化空间和通道校准模块
        self.spatial_calibrator = SpatialAttentionCalibrator(in_channels)
        self.channel_calibrator = ChannelAttentionCalibrator(in_channels, reduction_ratio)

    def forward(self, global_feat, target_feat):
        """
        注意力修正网络前向传播
        Args:
            global_feat: 全局特征图, shape [B, C, H, W]
            target_feat: 待校准特征图, shape [B, C, H, W]
        Returns:
            final_calibrated: 最终校准后的特征图, shape [B, C, H, W]
        """
        # 1. 分别进行空间和通道校准
        spatial_calibrated = self.spatial_calibrator(global_feat, target_feat)
        channel_calibrated = self.channel_calibrator(global_feat, target_feat)

        # 2. 融合两种校准结果
        final_calibrated = spatial_calibrated + channel_calibrated

        return final_calibrated


# 测试代码
if __name__ == "__main__":
    # 模拟输入：批量大小=2，通道数=64，特征图尺寸=32x32
    batch_size, in_channels, height, width = 2, 2048, 24, 12
    global_feat = torch.randn(batch_size, in_channels, height, width)
    target_feat = torch.randn(batch_size, in_channels, height, width)

    # 初始化修正网络
    acn = AttentionCorrectionNetwork(in_channels=in_channels)

    # 前向传播
    output = acn(global_feat, target_feat)

    # 打印输出形状（验证是否和输入一致）
    print(f"输入特征图形状: {target_feat.shape}")
    print(f"输出校准特征图形状: {output.shape}")
    print(f"可学习参数 alpha (空间校准): {acn.spatial_calibrator.alpha.item():.4f}")
    print(f"可学习参数 beta (通道校准): {acn.channel_calibrator.beta.item():.4f}")
