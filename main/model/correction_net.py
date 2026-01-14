from multiprocessing import reduction

import torch
import torch.nn as nn
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

        f1 = self.to_q(feat)  # [B, mid, H, W]
        f1 = rearrange(f1, "b mid h w -> b mid (h w)")  # [B, mid, H*W]

        f2 = self.to_k(feat)  # [B, mid, H, W]
        f2 = rearrange(f2, "b mid h w -> b mid (h w)")  # [B, H*W, mid]

        attn = torch.einsum("b m i, b m j -> b i j", f1, f2)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)

        cam_flat = rearrange(cam, "b c h w -> b (h w) c")  # [B, C, H*W]
        cam_refined_flat = torch.einsum(" b i j, b j c -> b i c", attn, cam_flat)
        cam_refined = rearrange(cam_refined_flat, "b (h w) c -> b c h w", h=H, w=W)

        return self.alpha * cam_refined + cam


# class Channel_Attention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(1.0))

#     def forward(self, feat_map, cam_feat_map):
#         B, C, H, W = cam_feat_map.shape

#         # 特征变换 + rearrange维度展平（这部分和之前一致，无需修改）
#         q = self.to_q(feat_map)  # [B, mid, H, W]
#         q = rearrange(q, "b c h w -> b c (h w)")  # [B, mid, H*W] (b c j)

#         k = self.to_k(feat_map)  # [B, mid, H, W]
#         k = rearrange(k, "b c h w -> b (h w) c")  # [B, H*W, mid] (b i c)

#         v = rearrange(cam_feat_map, "b c h w -> b c (h w)")  # [B, C, H*W]

#         attn = torch.einsum("b i c, b c j -> b i j", k, q)  # [B, H*W, H*W]
#         attn = torch.softmax(attn, dim=-1)

#         refined_cam_feat_map = torch.einsum("b c i, b i j -> b c j", v, attn)
#         refined_cam_feat_map = rearrange(refined_cam_feat_map, "b c (h w) -> b c h w", h=H, w=W)

#         return self.alpha * refined_cam_feat_map + cam_feat_map


# # 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, feat_i_dim):
        super().__init__()
        self.spatial_attn = Spatial_Attention(feat_i_dim)
        # self.channel_attn = Channel_Attention()

    def forward(self, feat_map, cam_feat_map):
        # cam_feat_map = (self.spatial_attn(cam, feat) + self.channel_attn(cam, feat)) / 2
        cam_feat_map = self.spatial_attn(feat_map, cam_feat_map)
        # cam_feat_map = self.channel_attn(feat_map, cam_feat_map)
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
