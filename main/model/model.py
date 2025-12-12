import copy

import torch
import torch.nn as nn

from .gem_pool import GeneralizedMeanPoolingP
from .model_main_tool import Calibration, Interaction, Propagation
from .resnet import resnet50
from .resnet_ibn_a import resnet50_ibn_a


class ReIDNet(nn.Module):

    def __init__(self, config, n_class):
        super(ReIDNet, self).__init__()
        self.config = config

        BACKBONE_FEATURES_DIM = config.MODEL.BACKBONE_FEATURES_DIM
        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone(BACKBONE_TYPE, non_local_flag=config.MODEL.NON_LOCAL_FLAG)

        # ------------- Global -----------------------
        if config.DATASET.TRAIN_DATASET == "sysu_mm01":
            self.global_pool = GeneralizedMeanPoolingP()
            self.global_classifier = Classifier(2048, n_class)
        elif config.DATASET.TRAIN_DATASET == "reg_db":

            # ------------- Partialization -----------------------
            self.local_conv_list = nn.ModuleList()
            STRIPE_NUM = 6
            pool_dim = 2048
            local_conv_out_channels = 512
            for _ in range(STRIPE_NUM):
                conv_i = nn.Sequential(
                    nn.Conv2d(pool_dim, local_conv_out_channels, 1),
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True),
                )
                conv_i.apply(weights_init_kaiming)
                self.local_conv_list.append(conv_i)

            # ------------- Global -----------------------
            self.global_classifier = Classifier(STRIPE_NUM * local_conv_out_channels, n_class)
            self.global_l2norm = Normalize(2)

            # ------------- Local -----------------------
            self.local_classifier_list = nn.ModuleList()
            for _ in range(STRIPE_NUM):
                local_classifier_i = Classifier(local_conv_out_channels, n_class)
                self.local_classifier_list.append(local_classifier_i)

        # ------------- Interaction -----------------------
        self.interaction = Interaction()
        if config.DATASET.TRAIN_DATASET == "reg_db":
            self.interaction.apply(weights_init_kaiming)

        # ------------- Calibration -----------------------
        self.calibration = Calibration()
        self.calibration_pooling = GeneralizedMeanPoolingP()
        self.calibration_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)
        if config.DATASET.TRAIN_DATASET == "reg_db":
            self.calibration.apply(weights_init_kaiming)
            self.calibration_pooling.apply(weights_init_kaiming)
            self.calibration_classifier.apply(weights_init_classifier)

        # # ------------- Propagation -----------------------
        self.propagation = Propagation(T=4)
        if config.DATASET.TRAIN_DATASET == "reg_db":
            self.propagation.apply(weights_init_kaiming)

    def heatmap(self, x_vis, x_inf, modal):
        B, C, H, W = x_vis.shape
        backbone_feat_map = self.backbone(x_vis, x_inf, modal)
        return backbone_feat_map

    def forward(self, x_vis, x_inf, modal):
        B, C, H, W = x_vis.shape

        backbone_feat_map = self.backbone(x_vis, x_inf, modal)

        if self.training:
            return backbone_feat_map
        else:
            eval_feats = []
            if self.config.DATASET.TRAIN_DATASET == "sysu_mm01":
                # ------------- Global -----------------------
                global_feat = self.global_pool(backbone_feat_map).view(B, 2048)  # (B, 2048)
                global_bn_feat, global_cls_score = self.global_classifier(global_feat)
                eval_feats.append(global_bn_feat)
            elif self.config.DATASET.TRAIN_DATASET == "reg_db":
                # ------------- Partialization -----------------------
                STRIPE_NUM = 6
                local_feat_map_list = torch.chunk(backbone_feat_map, STRIPE_NUM, dim=2)
                local_feat_list = []
                for i in range(STRIPE_NUM):
                    local_feat_map_i = local_feat_map_list[i]
                    local_feat_map_i = local_feat_map_i.view(B, 2048, -1)
                    p = 10.0  # regDB: 10.0    SYSU: 3.0
                    local_feat_i = (torch.mean(local_feat_map_i**p, dim=-1) + 1e-12) ** (1 / p)
                    local_feat_i = self.local_conv_list[i](local_feat_i.view(B, -1, 1, 1)).view(B, -1)
                    local_feat_list.append(local_feat_i)

                # ----------- Global ------------
                global_feat = torch.cat(local_feat_list, dim=1)
                global_bn_feat = self.global_l2norm(global_feat)
                eval_feats.append(global_bn_feat)

            eval_feats = torch.cat(eval_feats, dim=1)
            return eval_feats


#############################################################
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class Classifier(nn.Module):
    """
    BN -> Classifier
    """

    def __init__(self, c_dim, pid_num):
        super(Classifier, self).__init__()
        self.pid_num = pid_num

        self.bottleneck = nn.BatchNorm1d(c_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(c_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features):
        bn_features = self.bottleneck(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class Backbone(nn.Module):
    def __init__(self, backbone_type, non_local_flag):
        super(Backbone, self).__init__()
        self.non_local_flag = non_local_flag

        resnet = None
        if backbone_type == "resnet50":
            resnet = resnet50(pretrained=True)
        elif backbone_type == "resnet50_ibn_a":
            resnet = resnet50_ibn_a(pretrained=True)

        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        # Backbone structure
        self.vis_specific_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.inf_specific_layer = copy.deepcopy(self.vis_specific_layer)

        self.layer1 = resnet.layer1  # 3 blocks
        self.layer2 = resnet.layer2  # 4 blocks
        self.layer3 = resnet.layer3  # 6 blocks
        self.layer4 = resnet.layer4  # 3 blocks

        self.NL_2 = nn.ModuleList([Non_local(512) for i in range(2)])
        self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])

    def _NL_forward_layer(self, x, layer, NL_modules):
        num_blocks = len(layer)
        nl_start_idx = num_blocks - len(NL_modules)  # 从倒数层开始插入
        nl_counter = 0
        for i, block in enumerate(layer):
            x = block(x)
            if i >= nl_start_idx:
                x = NL_modules[nl_counter](x)
                nl_counter += 1
        return x

    def forward(self, x_vis, x_inf, modal):
        if modal == "all":
            x_vis = self.vis_specific_layer(x_vis)
            x_inf = self.inf_specific_layer(x_inf)
            x = torch.cat([x_vis, x_inf], dim=0)
        elif modal == "vis":
            x_vis = self.vis_specific_layer(x_vis)
            x = x_vis
        elif modal == "inf":
            x_inf = self.inf_specific_layer(x_inf)
            x = x_inf

        out = self.layer1(x)
        if self.non_local_flag:
            out = self._NL_forward_layer(out, self.layer2, self.NL_2)
            out = self._NL_forward_layer(out, self.layer3, self.NL_3)
        else:
            out = self.layer2(out)
            out = self.layer3(out)
        out = self.layer4(out)

        return out


#############################################################
class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


#############################################################
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
