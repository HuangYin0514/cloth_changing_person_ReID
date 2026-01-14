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

    def forward(self, feat_map, cam_feat_map):
        B, C, H, W = cam_feat_map.shape

        q = self.to_q(feat_map)  # [B, mid, H_cam, W_cam]
        q = rearrange(q, "b c h w -> b (h w) c")  # [B, N, mid]，N=H_cam*W_cam

        k = self.to_k(feat_map)  # [B, mid, H_cam, W_cam]
        k = rearrange(k, "b c h w -> b c (h w)")  # [B, mid, N]

        v = rearrange(cam_feat_map, "b c h w -> b c (h w)")  # [B, C_cam, N]

        attn = torch.einsum("b i c, b c j -> b i j", q, k)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)

        refined_cam_feat_map = torch.einsum("b c i, b i j -> b c j", v, attn)
        refined_cam_feat_map = rearrange(refined_cam_feat_map, "b c (h w) -> b c h w", h=H, w=W)

        return self.alpha * refined_cam_feat_map + cam_feat_map


# class Channel_Attention(nn.Module):
#     def __init__(self, feat_channels):
#         super().__init__()
#         # 注意力权重系数，初始值设为0.1，避免初始时残差占比过高
#         self.alpha = nn.Parameter(torch.tensor(0.1))
#         self.feat_channels = feat_channels  # 记录通道数，用于维度校验

#     def forward(self, feat_map, cam_feat_map):
#         B, C, H, W = cam_feat_map.shape

#         # 1. 展平空间维度：[B, C, H, W] -> [B, C, N]
#         feat_flat = rearrange(feat_map, "b c h w -> b c (h w)")  # 参考特征 [B, C, N]
#         cam_feat_flat = rearrange(cam_feat_map, "b c h w -> b c (h w)")  # 目标特征 [B, C, N]

#         # 2. 计算通道注意力权重（通道间相似度）
#         attn_weight = torch.einsum("b c1 n, b c2 n -> b c1 c2", feat_flat, cam_feat_flat)  # [B, C, C]
#         # 对每个c1对应的c2维度做softmax，确保通道注意力权重归一化
#         attn_weight = torch.softmax(attn_weight, dim=-1)  # dim=2 对应c2维度（目标通道）

#         # 3. 特征加权：用注意力权重加权目标特征
#         refined_cam_feat_flat = torch.einsum("b c2 n, b c1 c2 -> b c1 n", cam_feat_flat, attn_weight)
#         refined_cam_feat_map = rearrange(refined_cam_feat_flat, "b c (h w) -> b c h w", h=H, w=W)

#         # 残差连接：注意力加权特征 + 原始特征
#         output = self.alpha * refined_cam_feat_map + cam_feat_map
#         return output


# 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, feat_i_dim):
        super().__init__()
        self.spatial_attn = Spatial_Attention(feat_i_dim)
        # self.channel_attn = ChannelAttentionRefinement()

    def forward(self, feat_map, cam_feat_map):
        # cam_feat_map = (self.spatial_attn(cam, feat) + self.channel_attn(cam, feat)) / 2
        cam_feat_map = self.spatial_attn(feat_map, cam_feat_map)
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
