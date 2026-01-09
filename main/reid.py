import numpy as np
from tqdm import trange


def evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, ltcc_cc_setting=False, max_rank=50):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in trange(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_clothid = q_clothids[q_idx]

        order = indices[q_idx]
        if ltcc_cc_setting:  # remove gallery samples that have the same pid and (camid or clothid) with query
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) | (g_pids[order] == q_pid) & (g_clothids[order] == q_clothid)
        else:  # remove gallery samples that have the same pid and camid with query
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    # assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    print("Error: all query identities do not appear in gallery")

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


# def compute_ap_cmc(index, good_index, junk_index):
#     """Compute AP and CMC for each sample"""
#     ap = 0
#     cmc = np.zeros(len(index))

#     # remove junk_index
#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]

#     # find good_index index
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     rows_good = np.argwhere(mask == True)
#     rows_good = rows_good.flatten()

#     cmc[rows_good[0] :] = 1.0
#     for i in range(ngood):
#         d_recall = 1.0 / ngood
#         precision = (i + 1) * 1.0 / (rows_good[i] + 1)
#         ap = ap + d_recall * precision

#     return ap, cmc


# def evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode="CC"):
#     """Compute CMC and mAP with clothes

#     Args:
#         distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
#         q_pids (numpy array): person IDs for query samples.
#         g_pids (numpy array): person IDs for gallery samples.
#         q_camids (numpy array): camera IDs for query samples.
#         g_camids (numpy array): camera IDs for gallery samples.
#         q_clothids (numpy array): clothes IDs for query samples.
#         g_clothids (numpy array): clothes IDs for gallery samples.
#         mode: 'CC' for clothes-changing; 'SC' for the same clothes.
#     """
#     assert mode in ["CC", "SC"]

#     num_q, num_g = distmat.shape
#     index = np.argsort(distmat, axis=1)  # from small to large

#     num_no_gt = 0  # num of query imgs without groundtruth
#     num_r1 = 0
#     CMC = np.zeros(len(g_pids))
#     AP = 0

#     for i in range(num_q):
#         # groundtruth index
#         query_index = np.argwhere(g_pids == q_pids[i])  # pid相同
#         camera_index = np.argwhere(g_camids == q_camids[i])  # camid相同
#         cloth_index = np.argwhere(g_clothids == q_clothids[i])  # clothid相同
#         good_index = np.setdiff1d(
#             query_index, camera_index, assume_unique=True
#         )  # query_index和camera_index差集，pid相同且camid不同；【assume_unique=True表示假设输入的两个数组本身已经是无重复元素】
#         if mode == "CC":
#             good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)  # pid相同且camid不同且clothid不同
#             # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
#             junk_index1 = np.intersect1d(query_index, camera_index)  # query_index和camera_index的交集，pid相同且camid相同
#             junk_index2 = np.intersect1d(query_index, cloth_index)  # pid相同且clothid相同 **********************
#             junk_index = np.union1d(junk_index1, junk_index2)  # junk_index1和junk_index2的并集，pid相同且camid相同或pid相同且clothid相同
#         if mode == "SC":
#             good_index = np.intersect1d(good_index, cloth_index)
#             # remove gallery samples that have the same (pid, camid) or
#             # (the same pid and different clothid) with query
#             junk_index1 = np.intersect1d(query_index, camera_index)
#             junk_index2 = np.setdiff1d(query_index, cloth_index)  # query_index和cloth_index差集， 剔除 “不同衣物” 的样本 **********************
#             junk_index = np.union1d(junk_index1, junk_index2)

#         if good_index.size == 0:
#             num_no_gt += 1
#             continue

#         ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
#         if CMC_tmp[0] == 1:
#             num_r1 += 1
#         CMC = CMC + CMC_tmp
#         AP += ap_tmp

#     if num_no_gt > 0:
#         print("{} query samples do not have groundtruth.".format(num_no_gt))

#     if (num_q - num_no_gt) != 0:
#         CMC = CMC / (num_q - num_no_gt)
#         mAP = AP / (num_q - num_no_gt)
#     else:
#         mAP = 0

#     return CMC, mAP


################################################################################################################
# 标准reid
################################################################################################################


class ReIDEvaluator:

    def __init__(self, mode):
        assert mode in ["inter-camera", "intra-camera", "all"]
        self.mode = mode

    def evaluate(self, distmat, q_pids, q_camids, g_pids, g_camids):
        # 排序
        # rank_results = np.argsort(distmat)[:, ::-1]
        rank_results = np.argsort(distmat, axis=1)  # from small to large

        APs, CMC = [], []
        for idx, data in enumerate(zip(rank_results, q_pids, q_camids)):
            a_rank, q_pid, q_camid = data
            ap, cmc = self.compute_AP(a_rank, q_pid, q_camid, g_pids, g_camids)
            APs.append(ap), CMC.append(cmc)

        MAP = np.array(APs).mean()
        min_len = min([len(cmc) for cmc in CMC])
        CMC = [cmc[:min_len] for cmc in CMC]
        CMC = np.mean(np.array(CMC), axis=0)

        return MAP, CMC

    def compute_AP(self, a_rank, query_pid, query_cid, gallery_pids, gallery_cids):

        if self.mode == "inter-camera":
            # 多摄像头目标追踪：在商场、园区等多摄像头覆盖区域，追踪一个人从摄像头 A 的画面移动到摄像头 B、C 的轨迹。
            # 有效正样本: 同 ID 且不同相机的样本
            # 无效样本: 同 ID 且同相机的样本
            junk_index_1 = self.in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_cid == gallery_cids))  # 同 ID 且同相机的样本
            junk_index_2 = np.argwhere(gallery_pids == -1)  # 无 ID 的样本
            junk_index = np.append(junk_index_1, junk_index_2)  # 将两类无效样本的索引合并
            index_wo_junk = self.notin1d(a_rank, junk_index)  # 在排序数组中排除无效索引
            good_index = self.in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_cid != gallery_cids))  # 同 ID 且不同相机的样本

        if self.mode == "intra-camera":
            # 单摄像头内的目标跟踪：比如在一个监控摄像头画面中，持续追踪某个人的移动轨迹。
            # 有效正样本: 同 ID（且同相机）的样本（排除自身）
            # 无效样本: 不同相机的样本
            junk_index_1 = np.argwhere(query_cid != gallery_cids)
            junk_index_2 = np.argwhere(gallery_pids == -1)
            junk_index = np.append(junk_index_1, junk_index_2)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = np.argwhere(query_pid == gallery_pids)
            self_junk = a_rank[0]
            index_wo_junk = np.delete(index_wo_junk, np.where(self_junk == index_wo_junk))
            good_index = np.delete(good_index, np.where(self_junk == good_index))

        if self.mode == "all":
            # 不排除
            junk_index = np.argwhere(gallery_pids == -1)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = np.argwhere(query_pid == gallery_pids)
            self_junk = a_rank[0]
            index_wo_junk = np.delete(index_wo_junk, np.where(self_junk == index_wo_junk))
            good_index = np.delete(good_index, np.where(self_junk == good_index))

        hit = np.in1d(index_wo_junk, good_index)
        index_hit = np.argwhere(hit == True).flatten()
        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index_wo_junk)])
        else:
            precision = []
            for i in range(len(index_hit)):
                precision.append(float(i + 1) / float((index_hit[i] + 1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index_wo_junk)])
            cmc[index_hit[0] :] = 1
        return AP, cmc

    def in1d(self, array1, array2, invert=False):
        # a中的元素在b中
        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]

    def notin1d(self, array1, array2):
        # a中不在b中的元素
        return self.in1d(array1, array2, invert=True)
