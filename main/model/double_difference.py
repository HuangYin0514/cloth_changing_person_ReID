import torch
import torch.nn as nn


class DoubleDifferenceModule(nn.Module):
    """
    实现双次差分+批归一化的特征提取模块
    对应公式：F = BN[V_la_L - BN(V_la_L - V_s_L)]
    """

    def __init__(self, in_channels):
        super(DoubleDifferenceModule, self).__init__()
        # 定义批归一化层（BN），输入通道数需与特征图一致
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, V_la_L, V_s_L):
        """
        前向传播：执行双次差分和BN操作
        Args:
            V_la_L (torch.Tensor): 细粒度特征分支的多尺度特征，shape=[B, C, H, W]
            V_s_L (torch.Tensor): 另一分支的多尺度特征，shape=[B, C, H, W]
        Returns:
            F (torch.Tensor): 提取后的关键特征，shape=[B, C, H, W]
        """
        # 第一次差分：V_la_L - V_s_L
        first_diff = V_la_L - V_s_L

        # 对第一次差分结果做BN操作
        bn_first_diff = self.bn(first_diff)

        # 第二次差分：V_la_L - BN(第一次差分结果)
        second_diff = V_la_L - bn_first_diff

        # 对第二次差分结果做BN操作，得到最终特征F
        F = self.bn(second_diff)

        return F


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 模拟输入特征：批次B=2，通道C=64，高H=32，宽W=32
    # V_la_L：细粒度特征分支（已剔除干扰信息）
    V_la_L = torch.randn(2, 64, 32, 32)
    # V_s_L：另一分支的多尺度特征
    V_s_L = torch.randn(2, 64, 32, 32)

    # 初始化模块（输入通道数需与特征通道数一致）
    diff_module = DoubleDifferenceModule(in_channels=64)

    # 执行特征提取
    F = diff_module(V_la_L, V_s_L)

    # 输出结果形状，验证维度正确性
    print(f"输入特征形状: {V_la_L.shape}")
    print(f"输出关键特征形状: {F.shape}")
