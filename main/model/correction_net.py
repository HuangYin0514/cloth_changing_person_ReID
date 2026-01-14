from multiprocessing import reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 需要导入einops库


class ParallelPolarizedSelfAttention(nn.Module):
    """
    https://arxiv.org/pdf/2107.00782

    """

    def __init__(self, channel=2048):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.01))

    def forward(self, global_feat_map, cam_feat_map):
        b, c, h, w = global_feat_map.size()

        # # Spatial-only Self-Attention
        # spatial_wv = self.sp_wv(global_feat_map)  # bs,c//2,h,w
        # spatial_wq = self.sp_wq(global_feat_map)  # bs,c//2,h,w
        # spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        # spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        # spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        # spatial_wq = self.softmax_spatial(spatial_wq)
        # spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        # spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        # spatial_out = self.alpha * spatial_weight * cam_feat_map + cam_feat_map

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(global_feat_map)  # bs,c//2,h,w
        channel_wq = self.ch_wq(global_feat_map)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)  # bs,c,1,1
        channel_out = self.beta * channel_weight * cam_feat_map + cam_feat_map

        out = channel_out
        return out


# # 3. 双重注意力模块
class Correction_Net(nn.Module):
    def __init__(self, channel=2048):
        super().__init__()
        # self.spatial_attn = Spatial_Attention(feat_i_dim)
        self.att = ParallelPolarizedSelfAttention()

    def forward(self, global_feat_map, cam_feat_map):
        cam_feat_map = self.att(global_feat_map, cam_feat_map)
        return cam_feat_map


# 测试代码
if __name__ == "__main__":
    input = torch.randn(2, 2048, 24, 12)
    net = Correction_Net(channel=2048)
    output = net(input, input)
    print(output.shape)
    print("✅ 模块运行成功！")
