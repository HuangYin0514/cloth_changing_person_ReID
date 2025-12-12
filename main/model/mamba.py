import copy
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

POOL_HEGHT = 6
POOL_WIDTH = 1


class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class Featmap_2_Patch(nn.Module):
    def __init__(self):
        super(Featmap_2_Patch, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d((POOL_HEGHT, POOL_WIDTH))

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        feat_patch = rearrange(self.pooling(feat_map), "b c h w -> b c (h w)")
        return feat_patch


class Patch_2_Featmap(nn.Module):
    def __init__(self):
        super(Patch_2_Featmap, self).__init__()

    def forward(self, feat_patch):
        B, C, L = feat_patch.shape
        feat_map = rearrange(feat_patch.squeeze(), "b c (h w)-> b c h w", h=POOL_HEGHT, w=POOL_WIDTH)
        feat_map = F.interpolate(feat_map, size=(18, 9), mode="nearest")
        return feat_map


class CS_MAMBA(nn.Module):
    def __init__(self, in_cdim=2048, d_model=256):
        super(CS_MAMBA, self).__init__()

        # Mamba
        self.norm_1 = nn.LayerNorm(in_cdim)
        self.featmap_2_patch = Featmap_2_Patch()
        self.mamba = Mamba(in_cdim=in_cdim, d_model=d_model)
        # self.patch_2_featmap = Patch_2_Featmap()
        self.norm_2 = nn.LayerNorm(in_cdim)

        self.attention = ChannelAttention(in_channels=in_cdim)

        # FFN
        self.ffn_vis = nn.Sequential(
            nn.Conv2d(in_cdim, in_cdim, 1, 1, 0),
            nn.BatchNorm2d(in_cdim),
            nn.ReLU(),
        )
        self.ffn_inf = copy.deepcopy(self.ffn_vis)

    def forward(self, vis_feat_map, inf_feat_map):
        B, C, H, W = vis_feat_map.shape

        vis_feat_patch = self.featmap_2_patch(vis_feat_map)  # [B, C, n_patch]
        inf_feat_patch = self.featmap_2_patch(inf_feat_map)  # [B, C, n_patch]

        vi_feat_patch = torch.stack((vis_feat_patch, inf_feat_patch), dim=3)  # [B, C, n_patch, 2]
        vi_feat_patch = vi_feat_patch.view(B, C, -1)  # [B, C, 2n_patch]

        # ---- Mamba ----
        vi_feat_patch = rearrange(vi_feat_patch, "B D L -> B L D")  # [B, 2n_patch, C]
        vi_feat_patch = self.mamba(self.norm_1(vi_feat_patch)) + vi_feat_patch  # [B, 2n_patch, C]
        vi_feat_patch = self.norm_2(vi_feat_patch)
        vi_feat_patch = rearrange(vi_feat_patch, "B L D -> B D L")  # [B, C, 2n_patch]

        # --- Attention ---
        vis_feat_patch = vi_feat_patch[:, :, 0::2]  # [B, C, n_patch]
        inf_feat_patch = vi_feat_patch[:, :, 1::2]
        vis_attention = self.attention(vis_feat_patch.view(B, C, -1, 1)).view(B, C, 1, 1)
        inf_attention = self.attention(inf_feat_patch.view(B, C, -1, 1)).view(B, C, 1, 1)

        # ---- FFN ----
        out_vis = self.ffn_vis(vis_attention * vis_feat_map)
        out_inf = self.ffn_inf(inf_attention * inf_feat_map)
        return out_vis, out_inf


class Mamba(nn.Module):
    def __init__(self, in_cdim=2048, d_model=96):
        super(Mamba, self).__init__()

        ssm_ratio = 1.0
        d_inner = int(ssm_ratio * d_model)
        kernel_size = 3

        self.in_proj = nn.Linear(in_cdim, d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=kernel_size,
            groups=d_inner,
            padding=(kernel_size - 1) // 2,
        )
        self.ssm = SSM(d_model=d_inner)
        self.out_proj = nn.Linear(d_inner, in_cdim, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        x = self.in_proj(x)  # [B, L, D]
        x = rearrange(x, "B L D -> B D L")
        x = self.conv1d(x)
        x = rearrange(x, "B D L -> B L D")
        x = F.silu(x)  # [B, L, D]
        y = self.ssm(x)  # [B, L, D]
        out = self.out_proj(y)  # [B, L, in_cdim]
        return out


class SSM(nn.Module):

    def __init__(
        self,
        d_model=96,
        d_state=16,
    ):
        super(SSM, self).__init__()

        ssm_ratio = 1.0
        self.d_inner = int(ssm_ratio * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        # A
        A = repeat(torch.arange(1, self.d_state + 1), "d_state -> d_inner d_state", d_inner=self.d_inner)  # Shape (d_inner, state_dim); Ex. [[1, 2, ... , 16], ... ]
        self.A_log = nn.Parameter(torch.log(A))

        # x is projected to delta_ori, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # delta_ori is projected to delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # D
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Ex. [[1, 1, ... , 1], ... ]

    def forward(self, x):
        B, L, D = x.shape

        # Step 0: Get A and D
        A_parameter = -torch.exp(self.A_log.float())  # Shape (D, d_state); Ex. [-[1, 2, ... , 16], ... ]
        D_parameter = self.D.float()

        # Step 1: Project x to delta_B_C
        delta_B_C = self.x_proj(x)  # [B, L, D + d_state * 2]
        (delta, B_parameter, C_parameter) = delta_B_C.split(
            split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1
        )  # delta: (B, L, dt_rank). B, C: (B, L, d_state)
        delta_parameter = F.softplus(self.dt_proj(delta))  # (B, L, D)

        y = self.selective_scan(x, delta_parameter, A_parameter, B_parameter, C_parameter, D_parameter)  # [B, L, D]

        return y

    def selective_scan(self, u, delta_parameter, A_parameter, B_parameter, C_parameter, D_parameter):
        B, L, D = u.shape

        # Step 1: Discretize continuous parameters (A, B)
        delta_A = torch.exp(einsum(delta_parameter, A_parameter, "B L D, D state_dim -> B L D state_dim"))
        delta_B_u = einsum(delta_parameter, B_parameter, u, "B L D, B L state_dim, B L D -> B L D state_dim")

        x = torch.zeros((B, D, self.d_state), device=delta_A.device)
        ys = []
        for i in range(L):
            x = delta_A[:, i] * x + delta_B_u[:, i]
            y = einsum(x, C_parameter[:, i, :], "B D state_dim, B state_dim -> B D")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # [B, L, D]

        y = y + u * D_parameter  # [B, L, D]

        return y


if __name__ == "__main__":
    inp_1 = torch.randn(2, 6, 2048)
    print("input.shape", inp_1.shape)
    model = SSM(d_model=2048)
    outputs = model(inp_1)
    print(outputs.shape)

    inp_1 = torch.randn(2, 6, 2048)
    print("input.shape", inp_1.shape)
    model = Mamba(in_cdim=2048, d_model=96)
    outputs = model(inp_1)
    print(outputs.shape)
