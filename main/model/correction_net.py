from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库

from .da_att import PAM_Module_v2

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

178 修改注意力 
https://arxiv.org/pdf/1809.02983
DANet github https://github.com/CASIA-IVA-Lab/DANet/blob/master/encoding/nn/da_att.py#L19

180 unclothe_feat_map = torch.clamp(backbone_feat_map - clothe_feat_map, min=0)

181  out = self.gamma * out + (1 - self.gamma) * x2

182 total_loss += correction_clothe_loss

183 proj_value = self.value_conv(x2).view(m_batchsize, -1, width * height)

"""


# # 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, channel=2048):
        super().__init__()
        self.pa = PAM_Module_v2(channel)

    def forward(self, global_feat_map, cam_feat_map):
        B, C, H, W = global_feat_map.shape
        refined_cam_feat_map = self.pa(global_feat_map, cam_feat_map)
        return refined_cam_feat_map


# 测试代码
if __name__ == "__main__":
    input = torch.randn(2, 2048, 24, 12)
    net = Correction_Net(channel=2048)
    output = net(input, input)
    print(output.shape)
    print("✅ 模块运行成功！")
