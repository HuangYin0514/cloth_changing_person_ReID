import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from .bn_neck import BN_Neck
from .gem_pool import GeneralizedMeanPoolingP
from .weights_init import weights_init_classifier


class Clothe_Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Clothe_Classifier, self).__init__()

        self.GLOBAL_DIM = 2048
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.global_classifier = Linear_Classifier(self.GLOBAL_DIM, pid_num)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        feat = self.global_pool(feat_map).view(B, self.GLOBAL_DIM)
        bn_feat = self.global_bn_neck(feat)
        cls_score = self.global_classifier(bn_feat)
        return cls_score


class Linear_Classifier(nn.Module):

    def __init__(self, c_dim, pid_num):
        super(Linear_Classifier, self).__init__()
        self.pid_num = pid_num

        self.classifier = nn.Linear(c_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, feat):
        cls_score = self.classifier(feat)
        return cls_score


class BNNeck_Classifier(nn.Module):
    """
    BN_Neck -> Classifier
    """

    def __init__(self, c_dim, pid_num):
        super(BNNeck_Classifier, self).__init__()
        self.pid_num = pid_num
        self.bn_neck = BN_Neck(c_dim)
        self.classifier = Linear_Classifier(c_dim, self.pid_num)

    def forward(self, feat):
        bn_feat = self.bn_neck(feat)
        cls_score = self.classifier(bn_feat)
        return cls_score


class CC_Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)
        return y


class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        w = self.weight

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)
