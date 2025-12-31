import copy

import torch
import torch.nn as nn
import util
from torch.nn import init

from .classifier import CC_Classifier, Classifier
from .gem_pool import GeneralizedMeanPoolingP
from .module_main import MaxAvgPooling
from .resnet import resnet50
from .resnet_ibn_a import resnet50_ibn_a
from .weights_init import weights_init_classifier, weights_init_kaiming


class ReID_Net(nn.Module):

    def __init__(self, config, n_class):
        super(ReID_Net, self).__init__()
        self.config = config

        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone(BACKBONE_TYPE)

        # ------------- Global -----------------------
        self.GLOBAL_DIM = 4096
        self.global_pool = MaxAvgPooling()
        self.global_bn = nn.BatchNorm1d(self.GLOBAL_DIM)
        init.normal_(self.global_bn.weight.data, 1.0, 0.02)
        init.constant_(self.global_bn.bias.data, 0.0)
        self.global_classifier = CC_Classifier(self.GLOBAL_DIM, n_class)

    def heatmap(self, img):
        B, C, H, W = img.shape
        backbone_feat_map = self.backbone(img)
        return backbone_feat_map

    def forward(self, img):
        B, C, H, W = img.shape

        # ------------- Global -----------------------
        backbone_feat_map = self.backbone(img)
        global_feat = self.global_pool(backbone_feat_map).view(B, self.GLOBAL_DIM)
        global_bn_feat = self.global_bn(global_feat)

        if self.training:
            return backbone_feat_map, global_feat, global_bn_feat
        else:
            eval_feat_meter = []
            # ------------- Global -----------------------
            eval_feat_meter.append(global_bn_feat)

            eval_feat = torch.cat(eval_feat_meter, dim=1)
            return eval_feat


#############################################################


class Backbone(nn.Module):
    def __init__(self, backbone_type):
        super(Backbone, self).__init__()

        resnet = None
        if backbone_type == "resnet50":
            resnet = resnet50(pretrained=True)
        if backbone_type == "resnet50_ibn_a":
            resnet = resnet50_ibn_a(pretrained=True)

        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        # Backbone structure
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1  # 3 blocks
        self.layer2 = resnet.layer2  # 4 blocks
        self.layer3 = resnet.layer3  # 6 blocks
        self.layer4 = resnet.layer4  # 3 blocks

    def forward(self, img):
        out = self.layer0(img)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
