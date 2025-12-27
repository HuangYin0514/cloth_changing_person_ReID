import torch
import torch.nn as nn
import torch.optim as optim
from loss import ClothesBasedAdversarialLoss, CosFaceLoss
from model import NormalizedClassifier


class Build_Clothe_BASE:

    def __init__(self, config, *args, **kwargs):
        super(Build_Clothe_BASE, self).__init__()
        self.build(config, *args, **kwargs)

    def build(self, config, global_dim, num_train_clothes, pid2clothes, device):
        self.clothe_classifier = NormalizedClassifier(global_dim, num_train_clothes).to(device)

        model_params_group = [{"params": self.clothe_classifier.parameters(), "lr": config.OPTIMIZER.LEARNING_RATE, "weight_decay": 5e-4, "momentum": 0.9}]
        self.optimizer = optim.Adam(model_params_group)

        self.pid2clothes = torch.from_numpy(pid2clothes).to(device)

        self.criterion_ce = nn.CrossEntropyLoss().to(device)
        self.criterion_cfl = CosFaceLoss(scale=16.0, margin=0).to(device)
        self.criterion_adv = ClothesBasedAdversarialLoss()
