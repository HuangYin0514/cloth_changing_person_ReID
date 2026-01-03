import torch
import torch.nn as nn

from .bn_neck import BN_Neck
from .cam import CAM
from .classifier import Linear_Classifier
from .gem_pool import GeneralizedMeanPoolingP
from .pool_attention import Pool_Attention
from .resnet import resnet50
from .resnet_ibn_a import resnet50_ibn_a


class Part_Module(nn.Module):
    def __init__(self, c_dim=2048, part_num=8, part_dim=256, pool_type="avg"):
        super(Part_Module, self).__init__()
        self.c_dim = c_dim
        self.part_num = part_num
        self.part_dim = part_dim
        self.pool_type = pool_type

        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        if self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.part_conv_list = nn.ModuleList()
        for i in range(self.part_num):
            conv_i = nn.Sequential(
                nn.Linear(self.c_dim, self.part_dim, bias=False),
                nn.BatchNorm1d(self.part_dim),
            )
            self.part_conv_list.append(conv_i)

    def forward(self, feat_map):
        B, C, H, W = feat_map.size()

        part_len = H // self.part_num
        part_feat_list = []
        for i in range(self.part_num):
            local_feat_map = feat_map[:, :, i * part_len : (i + 1) * part_len, :]
            local_feat = self.pool(local_feat_map).flatten(1)
            part_feat_i = self.part_conv_list[i](local_feat)
            part_feat_list.append(part_feat_i)

        return part_feat_list
