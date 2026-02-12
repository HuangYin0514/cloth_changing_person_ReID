import os
import random
import shutil

import cv2
import numpy as np
import torch
import util
from reid import evaluate_ltcc
from torch.nn import functional as F
from util import time_now

from .test import cosine_dist, euclidean_dist, get_data, get_distmat, test


def visualization(config, reid_net, train_loader, query_loader, gallery_loader, logger, device):
    # visualization_heatmap(config, reid_net, train_loader, device)  # Grad-CAM对训练集可视化 / 可选可见光图像/红外图像
    visualization_rank(config, reid_net, train_loader, query_loader, gallery_loader, logger, device)
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


def visualization_rank(config, reid_net, train_loader, query_loader, gallery_loader, logger, device, *args, **kwargs):
    print(time_now(), "Visualization_ranked_results start")
    reid_net.eval()
    if config.DATA.TRAIN_DATASET == "ltcc":
        with torch.no_grad():
            qf, q_pids, q_camids, q_clothids = get_data(query_loader, reid_net, device)
            gf, g_pids, g_camids, g_clothids = get_data(gallery_loader, reid_net, device)

        distmat = get_distmat(qf, gf, dist="cosine")
        # distmat = get_distmat(qf, gf, dist="euclidean")

        # mAP, CMC = ReIDEvaluator(mode=config.TEST.TEST_MODE).evaluate(distmat, q_pids, q_camids, g_pids, g_camids)  # 标准测试/性能低于服装专用的评估器
        # logger("SC mode, \t mAP: {:.4f}; \t Rank: {}.".format(mAP, CMC[0:20]))

        CMC_CC, mAP_CC = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, ltcc_cc_setting=True)
        logger("CC mode, \t mAP: {:.4f}%; \t R-1: {:.4f}%. \t Rank: {}.".format(mAP_CC * 100, CMC_CC[0] * 100, CMC_CC[0:20]))

    rank_core = Rank_Core(config)
    rank_core.__call__(distmat, [query_loader.dataset, gallery_loader.dataset])
    print(time_now(), "Visualization_ranked_results done.")
    return


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


class Rank_Core:
    def __init__(self, config):
        self.config = config

        self.GRID_SPACING = 10
        self.QUERY_EXTRA_SPACING = 20
        self.BW = 2  # border width
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)

        self.width = 128
        self.height = 256
        self.topk = 10
        self.data_type = "image"

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]

        self.ranked_dir = os.path.join(config.SAVE.OUTPUT_PATH, "rank/")
        if not os.path.exists(self.ranked_dir):
            os.makedirs(self.ranked_dir)
            print("Successfully make dirs: {}".format(dir))
        else:
            shutil.rmtree(self.ranked_dir)
            os.makedirs(self.ranked_dir)

    def visualize_ranked_results(self, distmat, dataset, data_type, width=128, height=256, save_dir="", topk=10):
        print("Visualizing top-{} ranks ...".format(topk))
        num_q, num_g = distmat.shape
        print("# query: {}\t # gallery: {}".format(num_q, num_g))
        print("Visualizing top-{} ranks ...".format(topk))

        query, gallery = dataset
        indices = np.argsort(distmat, axis=1)

        for q_idx in range(num_q):
            q_feat, q_pid, q_camid, q_cloid = query[q_idx]
            # qcamid = 0

            if data_type == "image":
                qimg = tensor_2_image(q_feat, self.IMAGENET_MEAN, self.IMAGENET_STD)
                qimg = cv2.resize(qimg, (width, height))
                qimg = cv2.copyMakeBorder(qimg, self.BW, self.BW, self.BW, self.BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                qimg = cv2.resize(qimg, (width, height))  # resize twice to ensure that the border width is consistent across images
                num_cols = topk + 1
                grid_img = 255 * np.ones((height, num_cols * width + topk * self.GRID_SPACING + self.QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
                grid_img[:, :width, :] = qimg

            rank_idx = 1
            matched_num = 0
            for g_idx in indices[q_idx, :]:
                g_feat, g_pid, g_camid, g_cloid = gallery[g_idx]
                # gcamid = 1
                invalid = (q_pid == g_pid) & (q_camid == g_camid)  # 常规模式
                # invalid = (q_pid == g_pid) & (q_camid == g_camid) & (q_cloid == g_cloid)  # 换衣模式
                if not invalid:
                    matched = g_pid == q_pid
                    # if matched and rank_idx == 1:  # 过滤, rank-1 错误的情况
                    #     continue

                    # if not matched: # ********** 寻找下一个正确匹配的行人 **********
                    # print("q_pid: {}, g_pid: {}".format(q_pid, g_pid))
                    # continue

                    if matched:
                        matched_num += 1

                    if data_type == "image":
                        border_color = self.GREEN if matched else self.RED
                        gimg = tensor_2_image(g_feat, self.IMAGENET_MEAN, self.IMAGENET_STD)
                        gimg = cv2.resize(gimg, (width, height))
                        gimg = cv2.copyMakeBorder(gimg, self.BW, self.BW, self.BW, self.BW, cv2.BORDER_CONSTANT, value=border_color)
                        gimg = cv2.resize(gimg, (width, height))
                        start = rank_idx * width + rank_idx * self.GRID_SPACING + self.QUERY_EXTRA_SPACING
                        end = (rank_idx + 1) * width + rank_idx * self.GRID_SPACING + self.QUERY_EXTRA_SPACING
                        grid_img[:, start:end, :] = gimg
                    rank_idx += 1

                    if rank_idx > topk:
                        break

            if data_type == "image":
                # if q_pid != 19:  # 查询特定的行人图像
                #     continue
                # if matched_num < 3:
                #     continue
                imname = str(q_pid) + "_" + str(random.randint(100000, 999999))
                cv2.imwrite(os.path.join(save_dir, imname + ".jpg"), grid_img)

            if (q_idx + 1) % 100 == 0:
                print("- done {}/{}".format(q_idx + 1, num_q))
                # break # 只可视化前100张图片

    def __call__(self, distmat, dataset):
        # model.eval()
        # classifier.eval()
        self.visualize_ranked_results(
            distmat,
            dataset,
            data_type=self.data_type,
            width=self.width,
            height=self.height,
            save_dir=self.ranked_dir,
            topk=self.topk,
        )
        # model.train()
        # classifier.train()


def tensor_2_image(image, IMAGENET_MEAN, IMAGENET_STD):
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m).clamp_(0, 1)
    img_np = np.uint8(np.floor(image.cpu().detach().numpy() * 255))
    img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)
    return img_np[:, :, ::-1]
