import argparse
import os

import numpy as np
import util
from PIL import Image

# 固定参数
FIX_IMAGE_WIDTH = 144
FIX_IMAGE_HEIGHT = 288
RGB_CAMERAS = ["cam1", "cam2", "cam4", "cam5"]
IR_CAMERAS = ["cam3", "cam6"]

RGB_IMG_FILE_NAME = "train_rgb_resized_img_288_144.npy"
RGB_LABEL_FILE_NAME = "train_rgb_resized_label_288_144.npy"
IR_IMG_FILE_NAME = "train_ir_resized_img_288_144.npy"
IR_LABEL_FILE_NAME = "train_ir_resized_label_288_144.npy"


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, default="/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/SYSU_MM01")
    # parser.add_argument("--output_path", type=str, default="/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/SYSU_MM01")
    parser.add_argument("--data_path", type=str, default="/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/SYSU_MM01_concise")
    parser.add_argument("--output_path", type=str, default="/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/SYSU_MM01_concise")
    args = parser.parse_args()
    return args


def load_ids(data_path):
    """加载 train_id.txt 和 val_id.txt，并合并"""

    def _read_ids(file_path):
        with open(file_path, "r") as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(",")]
            return ["%04d" % x for x in ids]

    file_path_train = os.path.join(data_path, "exp/train_id.txt")
    file_path_val = os.path.join(data_path, "exp/val_id.txt")

    id_train = _read_ids(file_path_train)
    id_val = _read_ids(file_path_val)

    return sorted(id_train + id_val)


def collect_image_paths(data_path, ids):
    """收集 RGB 和 IR 图像路径"""
    files_rgb, files_ir = [], []
    for pid in ids:
        for cam in RGB_CAMERAS:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                files_rgb.extend(sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if ".DS_Store" not in fname]))  # 跳过.DS_Store文件

        for cam in IR_CAMERAS:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                files_ir.extend(sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if ".DS_Store" not in fname]))

    return files_rgb, files_ir


def build_pid2label(files_ir):
    """根据 IR 图像建立 PID 到标签的映射"""
    pid_container = {int(path[-13:-9]) for path in files_ir}
    return {pid: label for label, pid in enumerate(pid_container)}


def read_and_resize_images(image_paths, pid2label):
    """读取并缩放图像，返回像素数组和标签"""
    imgs, labels = [], []
    for path in image_paths:

        img = Image.open(path).resize((FIX_IMAGE_WIDTH, FIX_IMAGE_HEIGHT), Image.LANCZOS)
        imgs.append(np.array(img))

        pid = int(path[-13:-9])
        labels.append(pid2label[pid])

    return np.array(imgs), np.array(labels)


def process_dataset(data_path, output_path):
    """主函数：加载 ID、收集路径、读取图像并保存"""
    ids = load_ids(data_path)
    files_rgb, files_ir = collect_image_paths(data_path, ids)
    pid2label = build_pid2label(files_ir)
    # for fr in files_rgb:
    #     print(fr)
    # for fr in files_ir:
    #     print(fr)
    util.make_dirs(output_path)

    # RGB
    rgb_imgs, rgb_labels = read_and_resize_images(files_rgb, pid2label)
    np.save(os.path.join(output_path, RGB_IMG_FILE_NAME), rgb_imgs)
    np.save(os.path.join(output_path, RGB_LABEL_FILE_NAME), rgb_labels)

    # IR
    ir_imgs, ir_labels = read_and_resize_images(files_ir, pid2label)
    np.save(os.path.join(output_path, IR_IMG_FILE_NAME), ir_imgs)
    np.save(os.path.join(output_path, IR_LABEL_FILE_NAME), ir_labels)

    print("处理完成！RGB 图像：{}，IR 图像：{}".format(len(rgb_imgs), len(ir_imgs)))


if __name__ == "__main__":
    args = get_args()
    process_dataset(data_path=args.data_path, output_path=args.output_path)
