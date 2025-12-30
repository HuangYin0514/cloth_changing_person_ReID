import copy
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from . import image_transforms as T


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_mask_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def tensor_2_image(image, IMAGENET_MEAN=[0.485, 0.456, 0.406], IMAGENET_STD=[0.229, 0.224, 0.225]):
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m).clamp_(0, 1)
    img_np = np.uint8(np.floor(image.cpu().detach().numpy() * 255))
    img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)
    return img_np[:, :, ::-1]


class ImageDataset_Img_Mask(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.transform_mask = T.Compose(
            [
                T.Resize((384, 192)),
                T.Convert_ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if "LTCC" in img_path:
            # 遮罩图像
            root_dir = osp.dirname(osp.dirname(osp.dirname(img_path)))
            file_name = osp.basename(img_path)
            mask_path = osp.join(root_dir, "processed", file_name)
            mask_img = read_mask_image(mask_path)
            if self.transform_mask is not None:
                mask_img = self.transform_mask(mask_img)
                mask_img = (
                    F.interpolate(
                        mask_img.unsqueeze(0).unsqueeze(0),
                        size=(24, 12),
                        mode="nearest",
                        align_corners=None,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            # # 构造黑衣图
            # img_np = np.asarray(img, dtype=np.uint8)
            # mask_img_np = np.asarray(mask_img, dtype=np.uint8)
            # black_img = img_np.copy()
            # black_img[np.isin(mask_img_np, [2, 3, 4, 5, 6, 7, 10, 11])] = 0
            # black_img = Image.fromarray(black_img, mode="RGB")
            # if self.transform is not None:
            #     trans_black_img = self.transform(black_img)

        # 测试画图
        # plot_and_save_multi_np_images(tensor_2_image(img), mask_img.cpu().numpy())

        return img, mask_img, pid, camid, clothes_id


def plot_and_save_multi_np_images(*imgs, file_format: str = "png"):
    import os
    import shutil  # 用于递归删除目录
    import uuid

    import matplotlib.pyplot as plt
    import numpy as np

    """
    将任意数量的NumPy数组格式图像绘制在同一张画布上，自动布局，保存到tmp目录（随机文件名）

    参数:
        *imgs: np.ndarray - 任意数量的图像NumPy数组（支持灰度图/彩色图）
        file_format: str - 保存的文件格式，默认png，可选jpg/jpeg等

    异常处理:
        若未传入任何图像，抛出ValueError提示
    """
    # 1. 校验输入：至少传入1张图像
    if len(imgs) == 0:
        raise ValueError("请至少传入一张图像数组！")

    # 2. 创建tmp目录（如果不存在）
    save_dir = "tmp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. 生成随机文件名（UUID唯一标识符）
    random_name = str(uuid.uuid4()).replace("-", "")
    save_name = f"{random_name}.{file_format}"

    # 4. 自动计算画布布局（优先按行排列，每行最多3列）
    num_imgs = len(imgs)
    cols = min(3, num_imgs)  # 每行最多3列，避免画布过宽
    rows = (num_imgs + cols - 1) // cols  # 向上取整计算行数（比如4张图：(4+3-1)//3=2行）

    # 5. 设置画布大小（根据行列数动态调整，保证每张图显示清晰）
    fig_width = cols * 4  # 每列宽度4英寸
    fig_height = rows * 4  # 每行高度4英寸
    plt.figure(figsize=(fig_width, fig_height))

    # 6. 遍历所有图像，依次绘制子图
    for idx, img in enumerate(imgs, start=1):
        plt.subplot(rows, cols, idx)  # 定位到第idx个子图

        # 适配灰度图/彩色图
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        # 设置子图标题（显示第N张图）+ 隐藏坐标轴
        plt.title(f"Image {idx}")
        plt.axis("off")

    # 7. 调整子图间距，避免重叠
    plt.tight_layout()

    # 8. 保存图像到tmp目录
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()  # 释放绘图资源

    print(f"共{num_imgs}张图像已保存至: {save_path}")
    return save_path
