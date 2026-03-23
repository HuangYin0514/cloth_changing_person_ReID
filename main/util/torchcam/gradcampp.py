import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


class GradCAMpp:
    """
    GradCAM++ 核心实现类
    改进点：通过二阶导数优化梯度权重，能更精准定位多目标区域
    """

    def __init__(self, model, target_layer):
        """
        初始化GradCAM++
        :param model: 预训练CNN模型（如ResNet、VGG）
        :param target_layer: 目标卷积层（需可视化的层）
        """
        self.model = model
        self.target_layer = target_layer

        # 存储目标层的特征图和梯度
        self.activations = None  # 前向传播特征图 (B, C, H, W)
        self.gradients = None  # 反向传播梯度 (B, C, H, W)

        # 注册钩子捕获特征图和梯度
        self._register_hooks()

        # 模型设为评估模式
        self.model.eval()

    def _register_hooks(self):
        """注册前向/反向钩子函数"""

        # 前向钩子：捕获目标层输出（特征图）
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # 反向钩子：捕获目标层梯度
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 为目标层注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        """
        生成GradCAM++热力图
        :param input_tensor: 输入张量 (batch_size, 3, H, W)
        :param target_class: 目标类别索引（None则用预测的最高概率类别）
        :return: 归一化后的CAM热力图 (H, W)
        """
        # 1. 前向传播获取模型输出
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 2. 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 3. 反向传播计算梯度（保留计算图，需二阶导数）
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 4. GradCAM++ 核心：计算优化后的权重
        grads = self.gradients  # (B, C, H, W)
        acts = self.activations  # (B, C, H, W)

        # 步骤1：计算梯度的二阶导数（dy/dA）
        grads_sq = grads**2  # 梯度平方
        grads_cu = grads**3  # 梯度立方

        # 步骤2：计算每个特征图的权重 alpha
        # alpha = (grads_sq) / (2*grads_sq + acts*grads_cu 的全局和)
        numerator = grads_sq
        denominator = 2 * grads_sq + (acts * grads_cu).sum(dim=(2, 3), keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)  # 避免除0

        # 步骤3：alpha 乘以 ReLU(梯度)，再求和得到权重
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        # 5. 加权求和特征图，生成CAM
        cam = (weights * acts).sum(dim=1, keepdim=True)  # torch.Size([B, 1, H, W])

        ##############
        # mean_vals = cam.mean(dim=(2, 3), keepdim=True)  # 异常点处理
        # cam[:, :, :3, :2] = mean_vals
        # cam[:, :, :3, 2:] = mean_vals
        # cam = F.interpolate(cam[:, :, 3:-3, 2:-2], size=cam.shape[2:], mode="bilinear", align_corners=False)

        cam[:, :, :2, -2:] = cam[:, :, -1, -1]
        ##############

        # 6. ReLU激活（只保留正贡献）+ 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 归一化到0-1

        ###############
        # plt.figure(figsize=(12, 5))

        # # 原始图片
        # plt.subplot(1, 3, 1)
        # plt.imshow(input_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])
        # plt.axis("off")
        # plt.title("Original Image")

        # # GradCAM++结果
        # plt.subplot(1, 3, 2)
        # plt.imshow(cam.squeeze().cpu().numpy())
        # plt.axis("off")
        # plt.title("GradCAM++ Result")

        # plt.subplot(1, 3, 3)
        # plt_resize_input_tensor = F.interpolate(input_tensor, size=cam.shape[2:], mode="bilinear", align_corners=False)
        # plt_resize_input_tensor = 0.5 * plt_resize_input_tensor + 0.5 * cam
        # plt.imshow(plt_resize_input_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])
        # plt.axis("off")
        # plt.title("GradCAM++ Result")

        # plt.show(block=True)  # 防止窗口一闪而过
        ###############

        # 7. 上采样到输入尺寸，去除batch维度
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        return cam


# -------------------------- 复用的辅助可视化函数 --------------------------
def reverse_normalize(tensor):
    """还原归一化的图片张量到0-1范围"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def visualize(img_tensor, cam_map, alpha=0.5):
    """将CAM热力图叠加到原始图片"""
    # 张量转PIL图片并归一化
    img = transforms.ToPILImage()(img_tensor)
    img = np.array(img) / 255.0

    # 热力图转彩色
    cam_map = plt.cm.jet(cam_map)[:, :, :3]

    # 叠加
    mixed = (1 - alpha) * img + alpha * cam_map
    mixed = np.clip(mixed * 255, 0, 255).astype(np.uint8)

    return mixed


# -------------------------- 测试代码（可直接运行） --------------------------
if __name__ == "__main__":
    # 1. 加载图片
    image_path = "/Users/drhy/Documents/projects/cloth_changing_person_ReID/可视化/v2/img/003_1_c2_015279.png"
    image = Image.open(image_path).convert("RGB")

    # 2. 预处理（ImageNet标准）
    transform = transforms.Compose([transforms.Resize((384, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    # 3. 加载模型 + 初始化GradCAM++
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[-1]  # ResNet50最后一个卷积层
    grad_cam_pp = GradCAMpp(model, target_layer)

    # 4. 生成GradCAM++热力图
    cam_map = grad_cam_pp(input_tensor)

    # 5. 可视化结果
    img = reverse_normalize(input_tensor.squeeze(0))
    cam_image = visualize(img, cam_map)

    # 6. 显示对比图
    plt.figure(figsize=(12, 5))

    # 原始图片
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")

    # GradCAM++结果
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.axis("off")
    plt.title("GradCAM++ Result")

    plt.show(block=True)  # 防止窗口一闪而过
