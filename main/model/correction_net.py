from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库

from .da_att import PAM_Module_v2


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
