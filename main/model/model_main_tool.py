import copy

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .mamba import CS_MAMBA
from .model_aux_tool import Gate_Fusion


class Interaction(nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()

        self.cs_mamba = CS_MAMBA(in_cdim=2048)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape

        vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)

        vis_information, inf_information = self.cs_mamba(vis_feat_map, inf_feat_map)

        # Fusion
        vis_feat_map = inf_information + vis_feat_map
        inf_feat_map = vis_information + inf_feat_map
        feat_map = torch.cat([vis_feat_map, inf_feat_map], dim=0)
        return feat_map


# class Interaction(nn.Module):

#     def __init__(self):
#         super(Interaction, self).__init__()

#         self.vis_add_inf = nn.Sequential(
#             nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2048),
#             nn.ReLU(inplace=True),
#         )
#         self.inf_add_vis = nn.Sequential(
#             nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2048),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, feat_map):
#         vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)
#         vis_feat_map = self.vis_add_inf(inf_feat_map) + vis_feat_map
#         inf_feat_map = self.inf_add_vis(vis_feat_map) + inf_feat_map
#         feat_map = torch.cat([vis_feat_map, inf_feat_map], dim=0)
#         return feat_map


class Calibration(nn.Module):

    def __init__(self):
        super(Calibration, self).__init__()

        c_dim = 2048
        self.vis_gate_calibration = Gate_Fusion(c_dim)
        self.inf_gate_calibration = Gate_Fusion(c_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(c_dim * 2, c_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_dim, c_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, feat_map, res_feat_map):
        vis_feat, inf_feat = torch.chunk(feat_map, 2, dim=0)
        res_vis_feat, res_inf_feat = torch.chunk(res_feat_map, 2, dim=0)
        vis_feat = self.vis_gate_calibration(vis_feat, res_vis_feat)
        inf_feat = self.inf_gate_calibration(inf_feat, res_inf_feat)

        # calibration_feat_map = torch.cat([vis_feat, inf_feat], dim=0)
        calibration_feat_map = torch.cat([vis_feat, inf_feat], dim=1)
        calibration_feat_map = self.fusion(calibration_feat_map)
        calibration_feat_map = torch.cat([calibration_feat_map, calibration_feat_map], dim=0)
        return calibration_feat_map


class Propagation(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T=4):
        super(Propagation, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        p_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        loss = self.kl(p_s, p_t)  # * (self.T ** 2)
        return loss
