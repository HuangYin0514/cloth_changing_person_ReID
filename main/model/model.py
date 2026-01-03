import torch
import torch.nn as nn

from .bn_neck import BN_Neck
from .cam import CAM
from .classifier import Linear_Classifier
from .gem_pool import GeneralizedMeanPoolingP
from .part_module import Part_Module
from .pool_attention import Pool_Attention
from .resnet import resnet50
from .resnet_ibn_a import resnet50_ibn_a


class ReID_Net(nn.Module):

    def __init__(self, config, pid_num):
        super(ReID_Net, self).__init__()
        self.config = config

        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone(BACKBONE_TYPE)

        # ------------- Global -----------------------
        self.GLOBAL_DIM = 2048
        self.global_pool = GeneralizedMeanPoolingP()
        self.global_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.global_classifier = Linear_Classifier(self.GLOBAL_DIM, pid_num)

        # ------------- Cloth cam position -----------------------
        self.clothe_cam_position = CAM()
        self.clothe_cam_pool = GeneralizedMeanPoolingP()
        self.clothe_cam_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.clothe_cam_classifier = Linear_Classifier(self.GLOBAL_DIM, pid_num)

        # ------------- Bacbone inside -----------------------
        self.backbone_l4_b0_pool = nn.AdaptiveMaxPool2d(1)
        self.backbone_l4_b0_neck = BN_Neck(self.GLOBAL_DIM)
        self.backbone_l4_b0_part_module = Part_Module(self.GLOBAL_DIM, 8, 256, pool_type="max")
        self.backbone_l4_b0_classifier = Linear_Classifier(self.GLOBAL_DIM + 256 * 8, pid_num)

        self.backbone_l4_b1_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_l4_b1_neck = BN_Neck(self.GLOBAL_DIM)
        self.backbone_l4_b1_part_module = Part_Module(self.GLOBAL_DIM, 8, 256, pool_type="avg")
        self.backbone_l4_b1_classifier = Linear_Classifier(self.GLOBAL_DIM + 256 * 8, pid_num)

    # def heatmap(self, img):
    #     B, C, H, W = img.shape
    #     backbone_feat_map = self.backbone(img)
    #     return backbone_feat_map

    def forward(self, img):
        B, C, H, W = img.shape

        # ------------- Global -----------------------
        backbone_feat_map, backbone_l4_b0_feat_map, backbone_l4_b1_feat_map = self.backbone(img)
        global_feat = self.global_pool(backbone_feat_map).view(B, self.GLOBAL_DIM)
        global_bn_feat = self.global_bn_neck(global_feat)

        if self.training:
            backbone_inside_feat_map = {
                "backbone_l4_b0_feat_map": backbone_l4_b0_feat_map,
                "backbone_l4_b1_feat_map": backbone_l4_b1_feat_map,
            }
            return backbone_feat_map, global_feat, global_bn_feat, backbone_inside_feat_map
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

        self.layer4_0 = self.layer4[0]
        self.layer4_1 = self.layer4[1]
        self.layer4_2 = self.layer4[2]

        self.pool_att_0 = Pool_Attention(pool_type="max", feature_dim=2048)
        self.pool_att_1 = Pool_Attention(pool_type="avg", feature_dim=2048)

    def forward(self, img):
        out = self.layer0(img)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        out = self.layer4_0(out)
        out = self.pool_att_0(out)
        res_l4_b0 = out
        out = self.layer4_1(out)
        out = self.pool_att_1(out)
        res_l4_b1 = out
        out = self.layer4_2(out)
        return out, res_l4_b0, res_l4_b1
