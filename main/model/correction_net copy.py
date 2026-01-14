import torch
import torch.nn as nn
from einops import rearrange


# 原代码实现
class AttentionRefineModuleOriginal(nn.Module):
    def __init__(self, feat_channels, mid_channels):
        super().__init__()
        self.conv_f1 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.conv_f2 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, cam, feat):
        B, C, H, W = cam.shape
        f1 = self.conv_f1(feat).flatten(2)  # [B, mid, H*W]
        f2 = self.conv_f2(feat).flatten(2).transpose(1, 2)  # [B, H*W, mid]
        attn = torch.matmul(f2, f1)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)
        cam_flat = cam.flatten(2)
        cam_refined = torch.matmul(cam_flat, attn).unflatten(2, (H, W))
        return self.alpha * cam_refined + cam


# 新代码实现（attn在前）
class AttentionRefineModuleNew(nn.Module):
    def __init__(self, feat_channels, mid_channels):
        super().__init__()
        self.conv_f1 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.conv_f2 = nn.Conv2d(feat_channels, mid_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, cam, feat):
        B, C, H, W = cam.shape

        f1 = self.conv_f1(feat)  # [B, mid, H, W]
        f1 = rearrange(f1, "b mid h w -> b mid (h w)")  # [B, mid, H*W]

        f2 = self.conv_f2(feat)  # [B, mid, H, W]
        f2 = rearrange(f2, "b mid h w -> b mid (h w)")  # [B, H*W, mid]

        attn = torch.einsum("b m i, b m j -> b i j", f1, f2)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)

        cam_flat = rearrange(cam, "b c h w -> b (h w) c")  # [B, C, H*W]
        cam_refined_flat = torch.einsum(" b i j, b j c -> b i c", attn, cam_flat)
        cam_refined = rearrange(cam_refined_flat, "b (h w) c -> b c h w", h=H, w=W)

        return self.alpha * cam_refined + cam


# 对比测试
if __name__ == "__main__":
    # 固定随机种子，保证可复现
    torch.manual_seed(42)

    # 初始化两个模块（参数完全相同）
    feat_ch, mid_ch = 256, 128
    model_original = AttentionRefineModuleOriginal(feat_ch, mid_ch)
    model_new = AttentionRefineModuleNew(feat_ch, mid_ch)

    # 确保两个模型参数完全一致
    model_new.conv_f1.weight = model_original.conv_f1.weight
    model_new.conv_f2.weight = model_original.conv_f2.weight
    model_new.alpha = model_original.alpha

    # 构造测试输入
    B, C, H, W = 2, 256, 32, 32
    cam = torch.randn(B, C, H, W)
    feat = torch.randn(B, C, H, W)

    # 前向传播
    with torch.no_grad():
        out_original = model_original(cam, feat)
        out_new = model_new(cam, feat)

    # 计算结果差异
    diff = torch.abs(out_original - out_new).sum()
    print(f"原代码输出 vs 新代码输出 的总差异值：{diff.item():.4f}")
    print(f"原代码输出均值：{out_original.mean().item():.4f}")
    print(f"新代码输出均值：{out_new.mean().item():.4f}")

    # 验证维度是否一致（仅维度一致，数值不一致）
    assert out_original.shape == out_new.shape, "维度不一致！"
    print("维度验证：输出维度一致")
