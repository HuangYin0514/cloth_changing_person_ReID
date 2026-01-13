import torch
import torch.nn as nn


class CBR(nn.Module):
    """
    CBR模块：Conv2d -> BatchNorm2d -> ReLU
    整合了卷积、批量归一化和ReLU激活的常用组合模块

    Args:
        in_channels (int): 输入特征图的通道数
        out_channels (int): 输出特征图的通道数
        kernel_size (int or tuple): 卷积核大小，默认3
        stride (int or tuple): 卷积步长，默认1
        padding (int or tuple): 卷积填充，默认1（保持特征图尺寸不变）
        bias (bool): 卷积层是否使用偏置，默认False（BatchNorm会抵消偏置的作用）
        inplace (bool): ReLU是否使用inplace操作，默认True（节省内存）
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, inplace=True):
        super(CBR, self).__init__()

        # 定义CBR模块的三层结构
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        """
        前向传播：依次执行卷积 -> 批量归一化 -> ReLU

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, in_channels, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, out_channels, H', W']
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 测试示例
if __name__ == "__main__":
    # 创建CBR模块实例：输入3通道，输出16通道，默认参数
    cbr_module = CBR(in_channels=3, out_channels=16)

    # 生成测试输入：batch_size=2, 3通道, 32x32的特征图
    test_input = torch.randn(2, 3, 32, 32)

    # 前向传播
    output = cbr_module(test_input)

    # 打印输入输出形状
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    # 打印模块结构
    print("\nCBR模块结构:")
    print(cbr_module)
