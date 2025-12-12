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
    model.load_state_dict(torch.load(model_path), strict=False)
    print("Successfully resume model from {}".format(model_path))
