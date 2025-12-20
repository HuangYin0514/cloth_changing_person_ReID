import os
import random
import shutil

import cv2
import numpy as np
import torch
import util
from torch.nn import functional as F
from util import time_now


def visualization(config, reid_net, train_loader, query_loader, gallery_loader, logger, device):
    visualization_heatmap(config, reid_net, train_loader, device)  # Grad-CAM对训练集可视化 / 可选可见光图像/红外图像
    # visualization_rank(config, net, data_loder, query_loader, gallery_loader, DEVICE)
    # visualization_tsne(config, base, loader)


def visualization_heatmap(config, reid_net, train_loader, device, *args, **kwargs):
    print(time_now(), "CAM start")
    reid_net.eval()
    heatmap_loader = train_loader
    heatmap_core = Heatmap_Core(config)
    with torch.no_grad():
        for index, data in enumerate(heatmap_loader):
            if index % 100 == 0:
                print(time_now(), "CAM: {}/{}".format(index, len(heatmap_loader)))
            img, pid, camid, clotheid = data
            img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)
            heatmap_core.__call__(reid_net, reid_net.global_classifier, img, pid, camid, clotheid, *args, **kwargs)
            # break
    print(time_now(), "CAM done.")


##########################################################
# Core
##########################################################
class Heatmap_Core:
    def __init__(self, config):
        super(Heatmap_Core, self).__init__()
        self.config = config

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.GRID_SPACING = 10

        self.actmap_dir = os.path.join(config.SAVE.OUTPUT_PATH, "actmap/")
        if not os.path.exists(self.actmap_dir):
            os.makedirs(self.actmap_dir)
            print("Successfully make dirs: {}".format(dir))
        else:
            shutil.rmtree(self.actmap_dir)
            os.makedirs(self.actmap_dir)

    def channel_fn(self, features_map):
        heatmaps = torch.abs(features_map)
        # max_channel_indices = torch.argmax(heatmaps, dim=1, keepdim=True)[0]
        # print(max_channel_indices, max_channel_indices.shape)
        # heatmaps = torch.max(heatmaps[:, 476 : 476 + 1, :, :], dim=1, keepdim=True)[0]
        heatmaps = torch.max(heatmaps, dim=1, keepdim=True)[0]
        heatmaps = heatmaps.squeeze()
        return heatmaps

    def cam_fn(self, features_map, classifier, pids):
        bs, c, h, w = features_map.shape
        classifier_params = [param for name, param in classifier.named_parameters()]
        heatmaps = torch.zeros((bs, h, w))
        for i in range(bs):
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmaps[i] = heatmap_i
        return heatmaps

    def actmap_fn(self, reid_net, classifier, img, pid, camid, clotheid, *args, **kwargs):
        _, _, height, width = img.shape
        features_map = reid_net.heatmap(img)
        bs, c, h, w = features_map.shape

        # Channel
        # heatmaps = self.channel_fn(features_map)
        # CAM
        heatmaps = self.cam_fn(features_map, classifier, pid)

        mean_vals = heatmaps.mean(dim=(1, 2), keepdim=True)  # 异常点处理
        heatmaps[:, :3, :3] = mean_vals

        heatmaps = heatmaps.view(bs, h * w)
        heatmaps = F.normalize(heatmaps, p=2, dim=1)
        heatmaps = heatmaps.view(bs, h, w)

        for j in range(bs):

            # Image
            img_i = img[j, ...]
            for t, m, s in zip(img_i, self.IMAGENET_MEAN, self.IMAGENET_STD):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img_i.cpu().detach().numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # Activation map
            am = heatmaps[j, ...].cpu().detach().numpy()
            # am = outputs[j, 2:-2:, 2:-2].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # 重叠图像
            overlapped = img_np * 0.5 + am * 0.5
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones((height, 3 * width + 2 * self.GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + self.GRID_SPACING : 2 * width + self.GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * self.GRID_SPACING :, :] = overlapped

            random_number = random.randint(100000, 999999)
            cv2.imwrite(
                os.path.join(self.actmap_dir, str(pid[j].item()) + "_" + str(camid[j].item()) + "_" + str(clotheid[j].item()) + "_" + "_" + str(random_number) + ".jpg"),
                grid_img,
            )

    def __call__(self, *args, **kwargs):
        # model.eval()
        # classifier.eval()
        self.actmap_fn(*args, **kwargs)
        # model.train()
        # classifier.train()


def tensor_2_image(image, IMAGENET_MEAN, IMAGENET_STD):
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m).clamp_(0, 1)
    img_np = np.uint8(np.floor(image.cpu().detach().numpy() * 255))
    img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)
    return img_np[:, :, ::-1]
