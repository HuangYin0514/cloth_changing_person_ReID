import os

import torch

from .make_dirs import make_dirs
from .os_walk import os_walk


def save_model(model, epoch, path_dir):
    make_dirs(path_dir)
    model_file_path = os.path.join(path_dir, "model_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_file_path)

    root, _, files = os_walk(path_dir)
    for file in files:
        if ".pth" not in file:
            files.remove(file)  # 移除非模型文件

    if len(files) > 1:
        file_iters = sorted([int(file.replace(".pth", "").split("_")[1]) for file in files], reverse=False)
        model_file_path = os.path.join(root, "model_{}.pth".format(file_iters[0]))
        os.remove(model_file_path)  # 移除非最新的模型文件


def resume_model(model, resume_epoch, path):
    model_path = os.path.join(path, "model_{}.pth".format(resume_epoch))
    # model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
    # model.load_state_dict(torch.load(model_path), strict=False)

    # 自动检测设备并加载模型权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # 获取当前模型的参数字典
    model_state_dict = model.state_dict()

    filtered_checkpoint = {}
    for key, value in checkpoint.items():
        # 如果当前模型有这个参数，且形状匹配，就保留
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            filtered_checkpoint[key] = value
        # 其他不匹配的参数直接跳过（比如新增/删除的层）
        else:
            print(f"跳过不匹配的参数: {key} | 检查点形状: {value.shape} | 模型形状: {model_state_dict[key].shape}")

    # 加载过滤后的参数
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.to(device)  # 将模型移到对应设备

    print("Successfully resume model from {}".format(model_path))
