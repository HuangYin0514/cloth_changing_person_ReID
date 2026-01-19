from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库

from .SelfAttention import ScaledDotProductAttention

"""
多头注意力：https://comate.baidu.com/zh/page/jizkw42dul8


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
0.16888 0.375

174 修改注意力
0.16163 0.38265

175 修改注意力  cam_feat_map_flat_T = self.pa(global_feat_map_flat_T, cam_feat_map_flat_T, cam_feat_map_flat_T)
0.17345 0.37755

176 修改注意力 cam_feat_map_flat_T = self.pa(cam_feat_map_flat_T, global_feat_map_flat_T, global_feat_map_flat_T)

177 修改注意力 cam_feat_map_flat_T = self.pa(cam_feat_map_flat_T, global_feat_map_flat_T, cam_feat_map_flat_T)


"""


# # 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, channel=2048):
        super().__init__()
        self.pa = ScaledDotProductAttention(channel, d_k=512, d_v=512, h=1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, global_feat_map, cam_feat_map):
        B, C, H, W = global_feat_map.shape
        res_cam_feat_map = cam_feat_map

        # 空间校准
        global_feat_map_flat_T = global_feat_map.view(B, C, -1).permute(0, 2, 1)  # B, H*W, C
        cam_feat_map_flat_T = cam_feat_map.view(B, C, -1).permute(0, 2, 1)  # B, H*W, C
        cam_feat_map_flat_T = self.pa(cam_feat_map_flat_T, global_feat_map_flat_T, cam_feat_map_flat_T)
        cam_feat_map = cam_feat_map_flat_T.permute(0, 2, 1).view(B, C, H, W)  # B, C, H, W

        refined_cam_feat_map = self.alpha * cam_feat_map + res_cam_feat_map
        return refined_cam_feat_map


# 测试代码
if __name__ == "__main__":
    input = torch.randn(2, 2048, 24, 12)
    net = Correction_Net(channel=2048)
    output = net(input, input)
    print(output.shape)
    print("✅ 模块运行成功！")
