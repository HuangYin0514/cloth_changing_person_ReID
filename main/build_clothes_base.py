import torch
import torch.nn as nn
import torch.optim as optim
from loss import CosFaceLoss
from model import Linear_Classifier, NormalizedClassifier


class Build_Clothe_BASE:

    def __init__(self, config, *args, **kwargs):
        super(Build_Clothe_BASE, self).__init__()
        self.build(config, *args, **kwargs)

    def build(self, config, global_dim, num_clothe_pids, pid2clothes, device):
        self.clothe_classifier = Linear_Classifier(global_dim, num_clothe_pids).to(device)

        model_params_group = [{"params": self.clothe_classifier.parameters(), "lr": config.OPTIMIZER.LEARNING_RATE, "weight_decay": 5e-4, "momentum": 0.9}]
        self.optimizer = optim.Adam(model_params_group)
