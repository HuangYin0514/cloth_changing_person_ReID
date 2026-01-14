from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库


class Spatial_Attention(nn.Module):
    def __init__(self, feat_channels, reduction_ratio=16):
        super().__init__()
        mid_channels = feat_channels // reduction_ratio
        self.conv_f1 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.conv_f2 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_map, cam_feat_map):
        B, C, H, W = feat_map.shape
        # 特征变换+展平
        f1 = self.conv_f1(feat_map).flatten(2)  # [B, mid, H*W]
        f2 = self.conv_f2(feat_map).flatten(2).transpose(1, 2)  # [B, H*W, mid]
        # 空间注意力矩阵
        attn = torch.bmm(f2, f1)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)  # 用torch.softmax彻底避免歧义
        # 应用注意力
        cam_feat_map_flat = cam_feat_map.flatten(2)
        cam_feat_map_refined = torch.bmm(cam_feat_map_flat, attn).unflatten(2, (H, W))
        return self.alpha * cam_feat_map_refined + cam


class Channel_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.scale = 2048**-0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, feat_map, cam_feat_map):
        B, C, H, W = cam_feat_map.shape

        # 1. 维度变换
        feat_flat = feat_map.view(B, C, H * W)  # [B, C, H*W]
        cam_flat = cam_feat_map.view(B, C, H * W)  # [B, C, H*W]

        # 2. 点积注意力计算（通道维度的自注意力）
        attn_scores = torch.bmm(cam_flat, feat_flat.transpose(1, 2)) * self.scale  # (B, C, H*W) * (B, H*W, C) = (B, C, C)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        # 3. 应用注意力权重到CAM特征
        refined_cam_flat = torch.bmm(attn_scores, feat_flat)  # [B, C, C] * [B, C, H*W] = [B, C, H*W]
        refined_cam_feat_map = refined_cam_flat.view(B, C, H, W)  # [B, C, H, W]

        # 4. 特征融合：注意力加权特征 + 原始CAM特征
        return self.alpha * refined_cam_feat_map + cam_feat_map


# # 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, feat_i_dim):
        super().__init__()
        # self.spatial_attn = Spatial_Attention(feat_i_dim)
        self.channel_attn = Channel_Attention()

    def forward(self, feat_map, cam_feat_map):
        # cam_feat_map = (self.spatial_attn(cam, feat) + self.channel_attn(cam, feat)) / 2
        # cam_feat_map = self.spatial_attn(feat_map, cam_feat_map)
        cam_feat_map = self.channel_attn(feat_map, cam_feat_map)
        return cam_feat_map


# 测试代码
if __name__ == "__main__":
    # 测试参数
    B, C_cam, C_feat, H, W = 2, 2048, 2048, 24, 12
    # 随机输入
    cam = torch.randn(B, C_cam, H, W)
    feat = torch.randn(B, C_feat, H, W)
    # 初始化模块
    dual_module = Correction_Net(C_feat)
    # 前向传播
    cam_refined = dual_module(feat, cam)
    # 输出结果
    print(f"原始CAM形状: {cam.shape}")
    print(f"修正后CAM形状: {cam_refined.shape}")
    print("✅ 模块运行成功！")
