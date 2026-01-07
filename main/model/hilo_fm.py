import torch
import torch.nn as nn


def _get_masks(h, w, N=None, linspaces=None, mode="square"):
    assert mode in ["square", "rhombus", "circle"], f"mode should be'square', 'rhombus', or 'circle', but got {mode}"
    assert not (N is None and linspaces is None), "either N or linspaces should be provided"

    h_freqs = torch.fft.fftfreq(h)
    h_freqs = torch.fft.fftshift(h_freqs)
    w_freqs = torch.fft.rfftfreq(w)
    hw_freqs = torch.meshgrid(h_freqs, w_freqs, indexing="ij")

    if linspaces is None:
        rs = torch.linspace(0, 1, N + 1)[1:-1].tolist()
    else:
        if isinstance(linspaces, (int, float)):
            rs = [linspaces]
        else:
            rs = linspaces

    if mode == "square":
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]

            mask = torch.zeros_like(hw_freqs[0])
            flag = (hw_freqs[0].abs() <= vi) & (hw_freqs[1].abs() <= vi)

            mask[flag] = 1.0
            if i:
                for j in range(i):
                    mask = mask - masks[j]
            masks.append(mask)

        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)
    elif mode == "rhombus":
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]

            mask = torch.zeros_like(hw_freqs[0])
            flag = (
                ((hw_freqs[0].abs() + hw_freqs[1].abs()) <= vi) & ((hw_freqs[0].abs() + hw_freqs[1].abs()) > vi_1)
                if i
                else ((hw_freqs[0].abs() + hw_freqs[1].abs()) <= vi)
            )
            mask[flag] = 1.0

            masks.append(mask)

            vi_1 = vi

        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)

    else:
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]

            mask = torch.zeros_like(hw_freqs[0])
            flag = (
                ((hw_freqs[0] ** 2 + hw_freqs[1] ** 2) <= vi**2) & ((hw_freqs[0] ** 2 + hw_freqs[1] ** 2) > vi_1**2)
                if i
                else ((hw_freqs[0] ** 2 + hw_freqs[1] ** 2) <= vi**2)
            )
            mask[flag] = 1.0

            masks.append(mask)

            vi_1 = vi

        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)

    return torch.stack(masks, dim=0)  # [N, H, W]


class OSBlock(nn.Module):
    def __init__(self, shape, num_scales=4):
        super().__init__()
        in_channels = shape[0]

        self.num_scales = num_scales
        mid_channels = in_channels // num_scales

        lightconv3x3 = lambda channels: nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))

        self.conv2 = nn.ModuleList([nn.Sequential(*[lightconv3x3(mid_channels) for _ in range(i + 1)]) for i in range(num_scales)])

        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)

        self.share_attn = nn.Sequential(nn.Conv1d(2, 1, 5, 1, 2), nn.Sigmoid())
        nn.init.zeros_(self.share_attn[-2].bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = 0.0
        for i in range(self.num_scales):
            x2_ = self.conv2[i](x1)
            w = self.share_attn(torch.cat([x2_.mean((2, 3)).view(x.size(0), 1, -1), x2_.amax((2, 3)).view(x.size(0), 1, -1)], dim=1)).view(x.size(0), -1, 1, 1)
            x2 = x2 + x2_ * w

        x3 = self.conv3(x2)
        return x3


# ---------------------- 用户提供的HiLo_FM类 ----------------------
class HiLo_FM(nn.Module):
    def __init__(self, shape, ratio=0.2):
        super(HiLo_FM, self).__init__()
        self.shape = shape
        self.ratio = ratio

        in_dims, h, w = shape

        self.bn = nn.BatchNorm2d(in_dims)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

        self.hig_os = OSBlock(shape, 4)

        self.conv2 = nn.Sequential(nn.Conv2d(in_dims, 2, 3, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.conv2[-2].bias)

        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm="ortho"), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm="ortho")

        mask = _get_masks(h, w, linspaces=ratio, mode="square")[0]
        self.h_start = int(torch.argmax(mask, dim=0)[0].item())
        self.h_crop = int(mask.sum(0)[0].item())
        self.w_crop = int(mask.sum(1).max().item())
        self.register_buffer("low_mask", mask)

        self.low_weights = nn.Parameter(torch.randn(in_dims, self.h_crop, self.w_crop, 2) * 0.02)

    def forward(self, x):
        x = x.to(torch.float32)
        x_fft = self.fft(x)
        x_low = (x_fft * self.low_mask).clone()
        x_low[..., self.h_start : self.h_start + self.h_crop, : self.w_crop] = x_fft[
            ..., self.h_start : self.h_start + self.h_crop, : self.w_crop
        ] * torch.view_as_complex(self.low_weights)

        x_hig = x_fft * (1.0 - self.low_mask)
        x_ = torch.cat([x_low, x_hig], dim=0)
        x_ = self.ifft(x_)

        x_low, x_hig = x_.chunk(2, dim=0)

        x_hig = self.hig_os(x_hig)

        sp_attn = self.conv2(x)
        x_ = x_low * sp_attn[:, :1] + x_hig * sp_attn[:, 1:]

        return x + self.bn(x_)


# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # 1. 定义输入形状：(通道数, 高度, 宽度) = (2048, 24, 9)
    input_shape = (2048, 24, 9)
    # 2. 实例化HiLo_FM模块（ratio用默认值0.2）
    hilo_fm = HiLo_FM(shape=input_shape)

    # 3. 创建测试输入：batch_size=1，形状为 (B, C, H, W) = (1, 2048, 24, 9)
    batch_size = 1
    test_input = torch.randn(batch_size, *input_shape)

    # 4. 前向传播
    with torch.no_grad():  # 测试时禁用梯度计算，提升速度
        test_output = hilo_fm(test_input)

    # 5. 验证输出维度是否正确（应和输入维度一致）
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")

    # # 6. 额外验证关键参数
    # print(f"\n低频mask形状: {hilo_fm.low_mask.shape}")
    # print(f"low_weights形状: {hilo_fm.low_weights.shape}")
    # print(f"h_start: {hilo_fm.h_start}, h_crop: {hilo_fm.h_crop}, w_crop: {hilo_fm.w_crop}")

    # # 验证维度匹配性
    # assert test_input.shape == test_output.shape, "输入输出维度不一致！"
    # print("\n✅ 测试通过：输入输出维度匹配，模块前向传播正常！")
