import torch.nn as nn

from .bn_neck import BN_Neck


class Pool_Attention(nn.Module):
    def __init__(self, pool_type="avg", reduction=16, feature_dim=-1):
        super().__init__()
        self.pool_type = pool_type
        self.feature_dim = feature_dim

        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        if self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.refinement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // reduction, self.feature_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        B, C, H, W = feat.shape
        g_feat = self.pool(feat).flatten(1)
        refined_feat = self.refinement(g_feat).unsqueeze(-1).unsqueeze(-1) * feat
        return refined_feat
