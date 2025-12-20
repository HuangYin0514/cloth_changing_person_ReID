import time

import numpy as np
import torch
from re_rank import re_ranking
from reid import ReIDEvaluator
from sklearn import metrics as sk_metrics
from torch.nn import functional as F
from tqdm import tqdm
from util import CatMeter, time_now


def get_data(dataset_loader, reid_net, device):
    with torch.no_grad():
        feats_meter, pids_meter, camids_meter = CatMeter(), CatMeter(), CatMeter()
        for batch_idx, data in enumerate(tqdm(dataset_loader)):
            img, pid, camid, clothes_id = data
            img, pid, camid, clothes_id = img.to(device), pid.to(device), camid.to(device), clothes_id.to(device)

            # 原始特征
            bn_features = reid_net(img)

            # 翻转特征
            flip_images = torch.flip(img, [3])
            flip_bn_features = reid_net(flip_images)
            bn_features = bn_features + flip_bn_features

            feats_meter.update(bn_features.data)
            pids_meter.update(pid)
            camids_meter.update(camid)

    feats = feats_meter.get_val_numpy()
    pids = pids_meter.get_val_numpy()
    camids = camids_meter.get_val_numpy()

    return feats, pids, camids


def cosine_dist(x, y):
    def normalize(x):
        norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
        return x / norm

    x = normalize(x)
    y = normalize(y)
    return np.matmul(x, y.transpose([1, 0]))


def euclidean_dist(x, y):
    return sk_metrics.pairwise.euclidean_distances(x, y)


def gaussian_kernel_function(x, y):
    similarity = np.exp(np.square((x - y)))
    return similarity


def get_distmat(qf, gf, dist="cosine"):
    distmat = None
    if dist == "cosine":
        distmat = cosine_dist(qf, gf)
    if dist == "euclidean":
        distmat = euclidean_dist(qf, gf)
    if dist == "gaussian":
        distmat = gaussian_kernel_function(qf, gf)
    return distmat


def test(config, reid_net, query_loader, gallery_loader, device, logger):
    reid_net.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = get_data(query_loader, reid_net, device)
        gf, g_pids, g_camids = get_data(gallery_loader, reid_net, device)

    distmat = get_distmat(qf, gf)

    if config.TEST.RE_RANK:
        logger("Using re_ranking technology...")
        distmat = re_ranking(torch.from_numpy(qf), torch.from_numpy(gf), k1=20, k2=6, lambda_value=0.3)

    mAP, CMC = ReIDEvaluator(mode=config.TEST.TEST_MODE).evaluate(
        distmat,
        q_pids,
        q_camids,
        g_pids,
        g_camids,
    )
    return mAP, CMC[0:20]
