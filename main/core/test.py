import time

import numpy as np
import torch
from reid import ReIDEvaluator, evaluate_ltcc
from sklearn import metrics as sk_metrics
from torch.nn import functional as F
from tqdm import tqdm
from util import CatMeter, re_ranking, time_now


def get_data(dataset_loader, reid_net, device):
    with torch.no_grad():
        feats_meter, pids_meter, camids_meter, clothesid_meter = CatMeter(), CatMeter(), CatMeter(), CatMeter()
        for batch_idx, data in enumerate(tqdm(dataset_loader)):
            img, pid, camid, clothesid = data
            img, pid, camid, clothesid = img.to(device), pid.to(device), camid.to(device), clothesid.to(device)

            # 原始特征
            bn_features = reid_net(img)

            # 翻转特征
            flip_images = torch.flip(img, [3])
            flip_bn_features = reid_net(flip_images)
            bn_features = bn_features + flip_bn_features
            # bn_features = F.normalize(bn_features, p=2, dim=1)

            feats_meter.update(bn_features.data)
            pids_meter.update(pid)
            camids_meter.update(camid)
            clothesid_meter.update(clothesid)

    feats = feats_meter.get_val_numpy()
    pids = pids_meter.get_val_numpy()
    camids = camids_meter.get_val_numpy()
    clothes_id = clothesid_meter.get_val_numpy()

    return feats, pids, camids, clothes_id


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
        qf, q_pids, q_camids, q_clothids = get_data(query_loader, reid_net, device)
        gf, g_pids, g_camids, g_clothids = get_data(gallery_loader, reid_net, device)

    distmat = get_distmat(qf, gf)

    if config.TEST.RE_RANK:
        logger("Using re_ranking technology...")
        distmat = re_ranking(torch.from_numpy(qf), torch.from_numpy(gf), k1=20, k2=6, lambda_value=0.3)

    # mAP, CMC = ReIDEvaluator(mode=config.TEST.TEST_MODE).evaluate(distmat, q_pids, q_camids, g_pids, g_camids)  # 标准测试
    CMC_SC, mAP_SC = evaluate_ltcc(-distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode="SC")
    logger("SC mode, \t mAP: {:.2f}; \t Rank: {}.".format(mAP_SC, CMC_SC[0:20]))
    CMC_CC, mAP_CC = evaluate_ltcc(-distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode="CC")
    logger("CC mode, \t mAP: {:.2f}; \t Rank: {}.".format(mAP_CC, CMC_CC[0:20]))
    return mAP_CC, CMC_CC[0:20]
