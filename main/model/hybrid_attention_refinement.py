import torch
import torch.nn as nn
import torch.nn.functional as F


class Hybrid_Attention_Refinement(nn.Module):
    """
    混合注意力（通道+空间）：用参考特征图修正待修正特征图
    输入：
        ref_feat: 参考特征图，shape=[B, C, H, W]
        src_feat: 待修正特征图，shape=[B, C, H, W]
    输出：
        refined_feat: 修正后的特征图，shape=[B, C, H, W]
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        # 1. 特征融合层：融合参考与待修正特征，学习交互信息
        self.concat_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

        # 2. 通道注意力模块（Channel Attention, CA）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化：[B,C,H,W]→[B,C,1,1]
            nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0),
            nn.Sigmoid(),  # 通道权重：[B,C,1,1]
        )

        # 3. 空间注意力模块（Spatial Attention, SA）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # 空间权重：[B,1,H,W]
        )

        # 4. 残差连接：保留原始特征，提升稳定性
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ref_feat, src_feat):
        # 步骤1：融合参考特征与待修正特征，提取交互信息
        concat_feat = torch.cat([ref_feat, src_feat], dim=1)  # [B,2C,H,W]
        fused_feat = self.concat_conv(concat_feat)  # [B,C,H,W]

        # 步骤2：通道注意力加权（基于融合特征）
        channel_weight = self.channel_attn(fused_feat)  # [B,C,1,1]
        channel_refined = fused_feat * channel_weight  # 通道维度修正

        # 步骤3：空间注意力加权（基于通道修正后的特征）
        spatial_weight = self.spatial_attn(channel_refined)  # [B,1,H,W]
        spatial_refined = channel_refined * spatial_weight  # 空间维度修正

        # 步骤4：残差连接，融合原始特征与双重修正特征
        residual_feat = self.residual_conv(src_feat)
        refined_feat = self.relu(spatial_refined + residual_feat)

        return refined_feat


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 模拟特征图：B=2, C=64, H=32, W=32
    ref_feat = torch.randn(2, 64, 32, 32)
    src_feat = torch.randn(2, 64, 32, 32)

    model = Hybrid_Attention_Refinement(in_channels=64)
    refined_feat = model(ref_feat, src_feat)

    print(f"参考特征图shape: {ref_feat.shape}")
    print(f"待修正特征图shape: {src_feat.shape}")
    print(f"修正后特征图shape: {refined_feat.shape}")
    print(f"通道权重shape: {model.channel_attn(model.concat_conv(torch.cat([ref_feat, src_feat], dim=1))).shape}")
    print(
        f"空间权重shape: {model.spatial_attn(model.channel_attn(model.concat_conv(torch.cat([ref_feat, src_feat], dim=1))) * model.concat_conv(torch.cat([ref_feat, src_feat], dim=1))).shape}"
    )
