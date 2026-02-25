import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


class GradCAM:
    """
    GradCAM 核心实现类
    功能：生成卷积神经网络的梯度加权类激活映射
    """

    def __init__(self, model, target_layer):
        """
        初始化GradCAM
        :param model: 预训练的CNN模型（如ResNet、VGG等）
        :param target_layer: 目标卷积层（需要可视化的层）
        """
        self.model = model
        self.target_layer = target_layer

        # 初始化存储变量
        self.activations = None  # 存储目标层的前向传播特征图
        self.gradients = None  # 存储目标层的反向传播梯度

        # 注册前向/反向钩子，捕获特征图和梯度
        self._register_hooks()

        # 设置模型为评估模式
        self.model.eval()

    def _register_hooks(self):
        """注册钩子函数，捕获目标层的特征图和梯度"""

        # 前向钩子：捕获目标层的输出（特征图）
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # 反向钩子：捕获目标层的梯度
        def backward_hook(module, grad_input, grad_output):
            # grad_output是一个元组，第一个元素是输出梯度
            self.gradients = grad_output[0].detach()

        # 为目标层注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        """
        生成GradCAM热力图
        :param input_tensor: 输入张量 (batch_size, 3, H, W)
        :param target_class: 目标类别索引，None则使用预测的最高概率类别
        :return: 归一化后的CAM热力图 (H, W)
        """
        # 1. 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 2. 确定目标类别（默认用预测的最高概率类别）
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 3. 反向传播，计算目标类别的梯度
        # 只保留目标类别的梯度，其他类别置0
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 4. 计算梯度的全局平均池化（GAP）
        # gradients shape: (B, C, H, W) -> (1, C, 1, 1)
        weights = F.adaptive_avg_pool2d(self.gradients, 1)

        # 5. 加权求和特征图，得到CAM
        # activations shape: (B, C, H, W)
        # weights shape: (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # 6. ReLU激活（只保留正贡献）
        cam = F.relu(cam)

        # 7. 归一化到0-1之间
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 加小值避免除0

        # 8. 上采样到原始输入尺寸（224x224）并去除batch维度
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)  # 匹配输入的H,W
        cam = cam.squeeze().cpu().numpy()

        return cam


# -------------------------- 辅助可视化函数（替代utils.visualize） --------------------------
def reverse_normalize(tensor):
    """将归一化的张量还原为原始像素范围（0-1）"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def visualize(img_tensor, cam_map, alpha=0.5):
    """
    将CAM热力图叠加到原始图片上
    :param img_tensor: 还原后的图片张量 (3, H, W)
    :param cam_map: GradCAM生成的热力图 (H, W)
    :param alpha: 热力图透明度
    :return: 叠加后的PIL图片
    """
    # 将张量转为PIL图片
    img = transforms.ToPILImage()(img_tensor)
    img = np.array(img) / 255.0  # 归一化到0-1

    # 将CAM热力图转为彩色图
    cam_map = plt.cm.jet(cam_map)[:, :, :3]  # 取RGB通道，去掉Alpha

    # 叠加图片和热力图
    mixed = (1 - alpha) * img + alpha * cam_map
    mixed = np.clip(mixed * 255, 0, 255).astype(np.uint8)  # 还原为0-255

    return mixed


# -------------------------- 测试代码（可直接运行） --------------------------
if __name__ == "__main__":
    # 1. 加载图片
    image_path = "/Users/drhy/Documents/projects/cloth_changing_person_ReID/可视化/v2/img/003_1_c2_015279.png"
    image = Image.open(image_path).convert("RGB")

    # 2. 预处理
    transform = transforms.Compose([transforms.Resize((384, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    # 3. 加载模型和初始化GradCAM
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[-1]  # ResNet50最后一个卷积层
    grad_cam = GradCAM(model, target_layer)

    # 4. 生成CAM热力图
    cam_map = grad_cam(input_tensor)

    # 5. 可视化
    img = reverse_normalize(input_tensor.squeeze(0))
    cam_image = visualize(img, cam_map)

    # 6. 显示结果
    plt.figure(figsize=(10, 5))

    # 显示原始图片
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")

    # 显示CAM热力图
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.axis("off")
    plt.title("GradCAM Result")

    plt.show(block=True)
