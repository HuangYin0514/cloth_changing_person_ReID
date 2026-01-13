import torch
import torch.nn as nn


# 1. 独立空间注意力分支
class SpatialAttentionRefinement(nn.Module):
    def __init__(self, feat_channels, mid_channels):
        super().__init__()
        self.conv_f1 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.conv_f2 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, cam, feat):
        B, C, H, W = cam.shape
        # 特征变换+展平
        f1 = self.conv_f1(feat).flatten(2)  # [B, mid, H*W]
        f2 = self.conv_f2(feat).flatten(2).transpose(1, 2)  # [B, H*W, mid]
        # 空间注意力矩阵
        attn = torch.matmul(f2, f1)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)  # 用torch.softmax彻底避免歧义
        # 应用注意力
        cam_flat = cam.flatten(2)
        cam_refined = torch.matmul(cam_flat, attn).unflatten(2, (H, W))
        return self.alpha * cam_refined


# 2. 独立通道注意力分支
class ChannelAttentionRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, cam, feat):
        B, C, H, W = cam.shape
        # 展平
        cam_flat = cam.flatten(2)  # [B, C, H*W]
        feat_flat = feat.flatten(2)  # [B, feat_C, H*W]
        # 通道注意力矩阵
        attn = torch.matmul(cam_flat, feat_flat.transpose(1, 2))  # [B, C, feat_C]
        attn = torch.softmax(attn, dim=-1)
        # 应用注意力
        cam_refined = torch.matmul(attn, feat_flat).unflatten(2, (H, W))
        return self.beta * cam_refined


# 3. 双重注意力模块
class AttentionCorrectionNetwork(nn.Module):
    def __init__(self, feat_channels, mid_channels):
        super().__init__()
        self.spatial_attn = SpatialAttentionRefinement(feat_channels, mid_channels)
        self.channel_attn = ChannelAttentionRefinement()

    def forward(self, cam, feat):
        return (self.spatial_attn(cam, feat) + self.channel_attn(cam, feat)) / 2


# 测试代码
if __name__ == "__main__":
    # 测试参数
    B, C_cam, C_feat, mid_C, H, W = 2, 2048, 2048, 2048 // 16, 24, 12
    # 随机输入
    cam = torch.randn(B, C_cam, H, W)
    feat = torch.randn(B, C_feat, H, W)
    # 初始化模块
    dual_module = AttentionCorrectionNetwork(C_feat, mid_C)
    # 前向传播
    cam_refined = dual_module(cam, feat)
    # 输出结果
    print(f"原始CAM形状: {cam.shape}")
    print(f"修正后CAM形状: {cam_refined.shape}")
    print("✅ 模块运行成功！")
