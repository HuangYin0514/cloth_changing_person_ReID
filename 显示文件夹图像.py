import glob
import os
import warnings

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")  # 忽略无关警告

# -------------------------- 配置参数 --------------------------
IMAGE_DIR = r"/kaggle/working/cloth_changing_person_ReID/main/results/outputs/actmap/"  # 替换成你的图片目录，比如 r"C:\Users\XXX\Pictures"
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]  # 支持的图片格式
MAX_IMAGES = 100  # 最多显示100张
GRID_ROWS = 10  # 显示网格的行数
GRID_COLS = 5  # 显示网格的列数


# -------------------------- 核心代码 --------------------------
def display_images_from_dir(image_dir, max_images=100):
    """
    从指定目录显示指定数量的图片
    """
    # 获取目录下所有图片文件
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext), recursive=False))
        # 大小写都匹配（比如JPG和jpg）
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper()), recursive=False))

    # 去重并限制数量
    image_paths = list(set(image_paths))[:max_images]
    num_images = len(image_paths)

    if num_images == 0:
        print(f"在目录 {image_dir} 中未找到图片文件")
        return

    print(f"找到 {num_images} 张图片，将显示前 {min(num_images, max_images)} 张")

    # 设置显示样式
    plt.rcParams["figure.figsize"] = (20, 20)  # 设置整体画布大小
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图间距

    # 创建子图网格并显示图片
    for idx, img_path in enumerate(image_paths):
        try:
            # 打开并显示图片
            img = Image.open(img_path)
            # 调整子图位置（处理不足100张的情况）
            ax = plt.subplot(GRID_ROWS, GRID_COLS, idx + 1)
            ax.imshow(img)
            ax.axis("off")  # 关闭坐标轴
            ax.set_title(f"{idx+1}", fontsize=8)  # 显示图片序号
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue

    plt.tight_layout()  # 自动调整布局
    plt.show()


# 执行显示函数
display_images_from_dir(IMAGE_DIR, MAX_IMAGES)
