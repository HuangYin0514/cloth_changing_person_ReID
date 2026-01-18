from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库


class SpatialAttentionRefinement(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        mid_dim = in_dim // 16
        self.conv_f1 = nn.Conv2d(in_dim, mid_dim, 1, bias=False)
        self.conv_f2 = nn.Conv2d(in_dim, mid_dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat_map, cam_feat_map):
        B, C, H, W = cam_feat_map.shape

        # 特征变换+展平
        f1 = self.conv_f1(feat_map).flatten(2).transpose(1, 2)  # [B, mid, H*W]
        f2 = self.conv_f2(feat_map).flatten(2)  # [B, H*W, mid]

        # 空间注意力矩阵
        attn = torch.matmul(f1, f2)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)  # 用torch.softmax彻底避免歧义

        # 应用注意力
        cam_flat = cam_feat_map.flatten(2)  # [B, C, H*W]
        cam_refined = torch.matmul(cam_flat, attn).unflatten(2, (H, W))  # [B, C, H, W]
        return self.alpha * cam_refined + cam_feat_map


# # 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, channel=2048):
        super().__init__()
        self.sar = SpatialAttentionRefinement(channel)

    def forward(self, global_feat_map, cam_feat_map):
        cam_feat_map = self.sar(global_feat_map, cam_feat_map)
        return cam_feat_map


# 测试代码
if __name__ == "__main__":
    input = torch.randn(2, 2048, 24, 12)
    net = Correction_Net(channel=2048)
    output = net(input, input)
    print(output.shape)
    print("✅ 模块运行成功！")
