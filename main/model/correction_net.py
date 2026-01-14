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
        B, C, H, W = cam_feat_map.shape

        cam_feat_map_flat = rearrange(cam_feat_map, "b c h w -> b c (h w)")
        cam_feat_map_flat_T = rearrange(cam_feat_map, "b c h w -> b (h w) c")
        feat_map_flat = rearrange(feat_map, "b c h w -> b c (h w)")
        feat_map_flat_T = rearrange(feat_map, "b c h w -> b (h w) c")

        # attn = torch.einsum("b i n, b n j -> b i j", cam_feat_map_flat, feat_map_flat_T)
        attn = torch.einsum("b i n, b n j -> b i j", feat_map_flat, cam_feat_map_flat_T)
        attn = torch.softmax(attn, dim=-1)

        # refined_cam_feat_map = torch.einsum("b i j, b j n -> b i n", attn, feat_map_flat)
        # refined_cam_feat_map = rearrange(refined_cam_feat_map, "b c (h w) -> b c h w", h=H, w=W)
        refined_cam_feat_map = torch.einsum("b n i, b i j -> b n j", cam_feat_map_flat_T, attn)
        refined_cam_feat_map = rearrange(refined_cam_feat_map, "b (h w) c -> b c h w", h=H, w=W)
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
