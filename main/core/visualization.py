import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import util
from eval_metrics import eval_regdb, eval_sysu
from torch.nn import functional as F
from util import time_now


def visualization(config, net, data_loder, train_loader, query_loader, gallery_loader, DEVICE):
    visualization_heatmap(config, net, train_loader, DEVICE)  # Grad-CAM对训练集可视化 / 可选可见光图像/红外图像
    # visualization_rank(config, net, data_loder, query_loader, gallery_loader, DEVICE)
    # visualization_tsne(config, base, loader)


def visualization_heatmap(config, net, train_loader, DEVICE, *args, **kwargs):
    print(time_now(), "CAM start")
    net.eval()
    heatmap_loader = train_loader
    heatmap_core = Heatmap_Core(config)
    with torch.no_grad():
        for index, data in enumerate(heatmap_loader):
            if index % 100 == 0:
                print(time_now(), "CAM: {}/{}".format(index, len(heatmap_loader)))
            vis_imgs, inf_imgs, vis_labels, inf_labels = data
            vis_imgs = vis_imgs.to(DEVICE)
            inf_imgs = inf_imgs.to(DEVICE)
            heatmap_core.__call__(vis_imgs, net, net.global_classifier, vis_labels, modal="vis", *args, **kwargs)
            heatmap_core.__call__(inf_imgs, net, net.global_classifier, vis_labels, modal="inf", *args, **kwargs)
            # break
    print(time_now(), "CAM done.")


def visualization_rank(config, net, data_loder, query_loader, gallery_loader, DEVICE, *args, **kwargs):
    print(time_now(), "Visualization_ranked_results start")
    net.eval()
    loaders = [query_loader, gallery_loader]
    # ------------------------------------------------
    if config.DATASET.TRAIN_DATASET == "sysu_mm01":
        query_feat = np.zeros((data_loder.N_query, 2048))
        gall_feat = np.zeros((data_loder.N_gallery, 2048))
    loaders = [data_loder.query_loader, data_loder.gallery_loader]
    print(util.time_now(), "Start extracting features...")
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            if config.DATASET.TRAIN_DATASET == "sysu_mm01":
                modal_map = {0: "inf", 1: "vis"}
                modal = modal_map.get(loader_id)
            elif config.DATASET.TRAIN_DATASET == "reg_db":
                if config.TEST.REG_DB_MODE == "T2V":
                    modal_map = {1: "vis", 0: "inf"}
                else:
                    modal_map = {1: "inf", 0: "vis"}
                modal = modal_map.get(loader_id)
            ptr = 0
            for data in loader:
                imgs, pids = data
                batch_num = imgs.size(0)
                imgs = imgs.to(DEVICE)

                bn_feat = net(imgs, imgs, modal)
                flip_imgs = torch.flip(imgs, [3])
                flip_bn_feat = net(flip_imgs, flip_imgs, modal)
                bn_feat = bn_feat + flip_bn_feat

                if loader_id == 0:
                    query_feat[ptr : ptr + batch_num, :] = bn_feat.detach().cpu().numpy()
                elif loader_id == 1:
                    gall_feat[ptr : ptr + batch_num, :] = bn_feat.detach().cpu().numpy()

                ptr = ptr + batch_num

    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    def cos_sim(x, y):
        def normalize(x):
            norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
            return x / norm

        x = normalize(x)
        y = normalize(y)
        return np.matmul(x, y.transpose([1, 0]))

    distmat = cos_sim(query_feat, gall_feat)

    cmc, mAP = None, None
    if config.DATASET.TRAIN_DATASET == "sysu_mm01":
        cmc, mAP, mINP = eval_sysu(
            -distmat,
            data_loder.query_label,
            data_loder.gallery_label,
            data_loder.query_cam,
            data_loder.gallery_cam,
        )
    elif config.DATASET.TRAIN_DATASET == "reg_db":
        cmc, mAP, mINP = eval_regdb(
            -distmat,
            data_loder.query_label,
            data_loder.gallery_label,
        )

    print("mAP: {:.2%}\t , CMC:{:.2%}".format(mAP, cmc[0]))

    # ------------------------------------------------
    # t_dir = os.path.join(config.output_path, "tmp")
    # if not os.path.exists(t_dir):
    #     os.makedirs(t_dir)
    #     print("Successfully make dirs: {}".format(dir))

    # torch.save(query_features, os.path.join(config.output_path, "tmp", "query_features" + ".pt"))
    # torch.save(gallery_features, os.path.join(config.output_path, "tmp", "gallery_features" + ".pt"))

    # query_features = torch.load(os.path.join(config.output_path, "tmp", "query_features" + ".pt"))
    # gallery_features = torch.load(os.path.join(config.output_path, "tmp", "gallery_features" + ".pt"))

    # ------------------------------------------------
    def cos_sim(x, y):
        def normalize(x):
            norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
            return x / norm

        x = normalize(x)
        y = normalize(y)
        return np.matmul(x, y.transpose([1, 0]))

    dist = cos_sim(query_feat, gall_feat)
    rank_core = Rank_Core(config)
    rank_core.__call__(dist, [loaders[0].dataset, loaders[1].dataset])
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

    def actmap_fn(self, images, model, classifier, pids, modal, *args, **kwargs):
        _, _, height, width = images.shape
        features_map = model.heatmap(images, images, modal)
        bs, c, h, w = features_map.shape

        # CAM
        classifier_params = [param for name, param in classifier.named_parameters()]
        heatmaps = torch.zeros((bs, h, w))
        for i in range(bs):
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmaps[i] = heatmap_i

        # Channel
        # heatmaps = torch.abs(features_map)
        # # max_channel_indices = torch.argmax(heatmaps, dim=1, keepdim=True)[0]
        # # print(max_channel_indices, max_channel_indices.shape)
        # # heatmaps = torch.max(heatmaps[:, 476 : 476 + 1, :, :], dim=1, keepdim=True)[0]
        # heatmaps = torch.max(heatmaps, dim=1, keepdim=True)[0]
        # heatmaps = heatmaps.squeeze()

        mean_vals = heatmaps.mean(dim=(1, 2), keepdim=True)  # 异常点处理
        heatmaps[:, :3, :3] = mean_vals

        heatmaps = heatmaps.view(bs, h * w)
        heatmaps = F.normalize(heatmaps, p=2, dim=1)
        heatmaps = heatmaps.view(bs, h, w)

        for j in range(bs):

            # Image
            img = images[j, ...]
            for t, m, s in zip(img, self.IMAGENET_MEAN, self.IMAGENET_STD):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.cpu().detach().numpy() * 255))
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
            cv2.imwrite(os.path.join(self.actmap_dir, str(pids[j].item()) + "_" + modal + "_" + str(random_number) + ".jpg"), grid_img)

    def __call__(self, images, model, classifier, pids, *args, **kwargs):
        # model.eval()
        # classifier.eval()
        self.actmap_fn(images, model, classifier, pids, *args, **kwargs)
        # model.train()
        # classifier.train()


class Rank_Core:
    def __init__(self, config):
        self.config = config

        self.GRID_SPACING = 10
        self.QUERY_EXTRA_SPACING = 90
        self.BW = 5  # border width
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
        indices = np.argsort(distmat)[:, ::-1]

        for q_idx in range(num_q):
            q_feat, qpid = query[q_idx]
            qcamid = 0

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
                g_feat, gpid = gallery[g_idx]
                gcamid = 1
                invalid = (qpid == gpid) & (qcamid == gcamid)
                if not invalid:
                    matched = gpid == qpid
                    # if matched and rank_idx == 1:  # 过滤, rank-1 错误的情况
                    #     continue
                    if not matched:
                        print("qpid: {}, gpid: {}".format(qpid, gpid))
                        continue

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
                # if qpid != 19:  # 查询特定的行人图像
                #     continue
                # if matched_num < 3:
                #     continue
                imname = str(qpid) + "_" + str(random.randint(100000, 999999))
                cv2.imwrite(os.path.join(save_dir, imname + ".jpg"), grid_img)

            if (q_idx + 1) % 100 == 0:
                print("- done {}/{}".format(q_idx + 1, num_q))
                break

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
