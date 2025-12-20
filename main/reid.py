import numpy as np
from tqdm import trange


class ReIDEvaluator:

    def __init__(self, mode):
        assert mode in ["inter-camera", "intra-camera", "all"]
        self.mode = mode

    def evaluate(self, distmat, q_pids, q_camids, g_pids, g_camids):
        # 排序
        rank_results = np.argsort(distmat)[:, ::-1]

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

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
