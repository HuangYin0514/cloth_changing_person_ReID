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
    def __init__(self, num_pid, num_part=6, c_dim=2048, part_dim=256, pool_type="avg"):
        super(Part_Module, self).__init__()
        self.num_part = num_part
        self.num_pid = num_pid
        self.num_part = num_part
        self.c_dim = c_dim
        self.part_dim = part_dim
        self.pool_type = pool_type

        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        if self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        if self.pool_type == "gem":
            self.pool = GeneralizedMeanPoolingP()

        self.part_conv_list = nn.ModuleList()
        self.classifier_list = nn.ModuleList()
        for i in range(self.num_part):
            conv_i = nn.Sequential(
                nn.Conv2d(2048, part_dim, 1, 1, 0),
                nn.BatchNorm2d(self.part_dim),
                nn.ReLU(inplace=True),
            )
            self.part_conv_list.append(conv_i)

            classifier_i = Linear_Classifier(part_dim, num_pid)
            self.classifier_list.append(classifier_i)

        self.classifier = Linear_Classifier(self.num_part * self.part_dim, self.num_pid)

    def forward(self, feat_map, global_bn_feat):
        B, C, H, W = feat_map.size()

        part_len = H // self.num_part
        part_feat_list = []
        part_cls_score_list = []
        for i in range(self.num_part):
            part_feat_map_i = feat_map[:, :, i * part_len : (i + 1) * part_len, :]
            part_feat_i = self.pool(part_feat_map_i)
            part_feat_i = self.part_conv_list[i](part_feat_i).view(B, self.part_dim)
            part_feat_list.append(part_feat_i)
            part_cls_score_i = self.classifier_list[i](part_feat_i)
            part_cls_score_list.append(part_cls_score_i)
        # part_feat = torch.cat(part_feat_list, dim=1)
        # part_feat = torch.cat([part_feat, global_bn_feat], dim=1)
        # part_cls_score = self.classifier(part_feat)

        return part_feat_list, part_cls_score_list
