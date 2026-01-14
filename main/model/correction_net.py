from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库


class Spatial_Attention(nn.Module):
    def __init__(self, feat_channels, reduction_ratio=16):
        super().__init__()
        mid_channels = feat_channels // reduction_ratio

        self.to_q = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.to_k = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, cam):
        B, C, H, W = cam.shape

        f1 = self.to_q(feat)
        f1_flat_T = rearrange(f1, "b c h w -> b (h w) c")

        f2 = self.to_k(feat)
        f2_flat = rearrange(f2, "b c h w -> b c (h w)")

        attn = torch.einsum("b i c, b c j -> b i j", f1_flat_T, f2_flat)
        attn = torch.softmax(attn, dim=-1)

        cam_flat = rearrange(cam, "b c h w -> b c (h w)")
        cam_refined_flat = torch.einsum("b c i, b i j -> b c j", cam_flat, attn)
        cam_refined = rearrange(cam_refined_flat, "b c (h w) -> b c h w", h=H, w=W)

        return self.alpha * cam_refined + cam


class Channel_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_map, cam_feat_map):
        """
        点积注意力计算
        Args:
            feat_map: 原始特征图，形状 [B, C, H, W]
            cam_feat_map: CAM特征图，形状 [B, C, H, W]
        Returns:
            融合后的特征图，形状 [B, C, H, W]
        """
        B, C, H, W = cam_feat_map.shape

        # 1. 维度变换：将空间维度展平，便于计算点积 [B, C, H*W]
        feat_flat = feat_map.view(B, C, H * W)  # 原始特征展平
        cam_flat = cam_feat_map.view(B, C, H * W)  # CAM特征展平

        # 2. 点积注意力计算（通道维度的自注意力）
        # 计算注意力分数：(B, C, H*W) * (B, H*W, C) = (B, C, C)
        attn_scores = torch.bmm(feat_flat, cam_flat.transpose(1, 2))
        # 归一化注意力分数（softmax在通道维度）
        attn_scores = F.softmax(attn_scores / torch.sqrt(torch.tensor(C, dtype=torch.float32)), dim=-1)

        # 3. 应用注意力权重到CAM特征 [B, C, C] * [B, C, H*W] = [B, C, H*W]
        refined_cam_flat = torch.bmm(attn_scores, cam_flat)
        # 恢复原始空间维度 [B, C, H, W]
        refined_cam_feat_map = refined_cam_flat.view(B, C, H, W)

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
