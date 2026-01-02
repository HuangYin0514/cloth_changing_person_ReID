from torch import nn

from .weights_init import weights_init_kaiming


class BN_Neck(nn.Module):

    def __init__(self, c_dim):
        super(BN_Neck, self).__init__()

        self.bn_neck = nn.BatchNorm1d(c_dim)
        self.bn_neck.bias.requires_grad_(False)  # no shift
        self.bn_neck.apply(weights_init_kaiming)

    def forward(self, feat):
        bn_feat = self.bn_neck(feat)
        return bn_feat
