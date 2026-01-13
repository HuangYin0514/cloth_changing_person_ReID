from multiprocessing import reduction

import torch
import torch.nn as nn
from einops import rearrange


class Dot_Product_Attention(nn.Module):
    """
    feat_i_dim [B, C, H, W]
    feat_j_dim [B, C, H, W]
    """

    def __init__(self, feat_i_dim, reduction=16):
        super().__init__()
        inner_dim = feat_i_dim // reduction

        self.to_q = nn.Conv2d(feat_i_dim, inner_dim, 1, 1, 0, bias=False)
        self.to_k = nn.Conv2d(feat_i_dim, inner_dim, 1, 1, 0, bias=False)

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_map_i, feat_map_j):
        B, C, H, W = feat_map_i.shape

        q = self.to_q(feat_map_i)
        q = rearrange(q, "b d H W -> b (H W) d")

        k = self.to_k(feat_map_i)
        k = rearrange(k, "b d H W -> b (H W) d")

        v = rearrange(feat_map_j, "b d H W -> b (H W) d")

        dots = torch.einsum("b i d, b j d -> b i j", q, k)
        attn = dots.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "b (H W) d -> b d H W", H=H, W=W)

        return self.alpha * out + feat_map_j


# 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, feat_i_dim):
        super().__init__()
        self.spatial_attn = Dot_Product_Attention(feat_i_dim)
        # self.channel_attn = ChannelAttentionRefinement()

    def forward(self, feat_map_i, feat_map_j):
        # cam_feat_map = (self.spatial_attn(cam, feat) + self.channel_attn(cam, feat)) / 2
        cam_feat_map = self.spatial_attn(feat_map_i, feat_map_j)
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
