import torch
from sympy import im
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from .weights_init import weights_init_classifier, weights_init_kaiming


class BN_Neck(nn.Module):

    def __init__(self, c_dim):
        super(BN_Neck, self).__init__()

        self.bn_neck = nn.BatchNorm1d(c_dim)
        self.bn_neck.bias.requires_grad_(False)  # no shift
        self.bn_neck.apply(weights_init_kaiming)

    def forward(self, feat):
        bn_feat = self.bn_neck(feat)
        return bn_feat


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
        self.Learn_classifier = Linear_Classifier(c_dim, self.pid_num)

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
