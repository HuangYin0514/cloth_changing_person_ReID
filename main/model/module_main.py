import torch
import torch.nn as nn
import util

from .classifier import BN_Neck, Linear_Classifier
from .gem_pool import GeneralizedMeanPoolingP


class Empty_Module(nn.Module):
    def __init__(self):
        super(Empty_Module, self).__init__()

    def forward(self, x):
        return


class Clothe_Classifier_Net(nn.Module):
    def __init__(self, pid_num):
        super(Clothe_Classifier_Net, self).__init__()

        self.GLOBAL_DIM = 2048
        self.global_pool = GeneralizedMeanPoolingP()
        self.global_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.global_classifier = Linear_Classifier(self.GLOBAL_DIM, pid_num)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        feat = self.global_pool(feat_map).view(B, self.GLOBAL_DIM)
        bn_feat = self.global_bn_neck(feat)
        cls_score = self.global_classifier(bn_feat)
        return cls_score
