import torch
import torch.nn as nn


class CAM(nn.Module):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, features_map, pids, classifier):
        size, c, h, w = features_map.size()

        # 提前提取分类器最后一层的权重
        with torch.no_grad():
            classifier_weight = list(classifier.parameters())[-1]  # shape: [num_classes, c]
            selected_weights = classifier_weight[pids]  # shape: [batch_size, c]

            # 计算热力图
            features_map_flat = features_map.view(size, c, h * w)  # [batch, c, h*w]
            heatmaps = torch.bmm(selected_weights.unsqueeze(1), features_map_flat)  # [batch, 1, h*w]
            heatmaps = heatmaps.view(size, h, w)

            # 归一化
            heatmaps_min = heatmaps.view(size, -1).min(dim=1, keepdim=True)[0].view(size, 1, 1)
            heatmaps_max = heatmaps.view(size, -1).max(dim=1, keepdim=True)[0].view(size, 1, 1)
            heatmaps = (heatmaps - heatmaps_min) / (heatmaps_max - heatmaps_min + 1e-6)  # 避免除以0

        # 应用热力图
        features_map_out = features_map * heatmaps.unsqueeze(1)

        return features_map_out
