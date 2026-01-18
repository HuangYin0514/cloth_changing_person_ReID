from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库

"""
167 极注意力 spatial_out = spatial_weight * cam_feat_map
Best model is: epoch: 79, mAP: 16.2357%, rank1: 36.2245%.

168 极注意力 spatial_wq = self.sp_wq(cam_feat_map)  # bs,c//2,h,w
0.16129 0.35714

169 空间注意力 self.alpha * cam_refined + cam_feat_map
0.1705 0.38776

170 空间注意力 0.01 * self.alpha * cam_refined + cam_feat_map
0.17722 0.375   性能较好

171 空间注意力  0.01 * cam_refined + cam_feat_map
0.15465 0.34184

173 空间注意力  0.001 * self.alpha * cam_refined + cam_feat_map

"""
# 171 0.01 * cam_refined + cam_feat_map
# 172 in_dim // 4 -> 对比 170


class SpatialAttentionRefinement(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        mid_dim = in_dim // 4
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
        return 0.001 * self.alpha * cam_refined + cam_feat_map


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
