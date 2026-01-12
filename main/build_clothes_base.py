import torch
import torch.nn as nn
import torch.optim as optim
from loss import CrossEntropyLabelSmooth
from model import Clothe_Classifier


class Build_Clothe_BASE:

    def __init__(self, config, *args, **kwargs):
        super(Build_Clothe_BASE, self).__init__()
        self.build(config, *args, **kwargs)

    def build(self, config, num_clothe_pids, pid2clothes, device):
        self.clothe_classifier = Clothe_Classifier(num_clothe_pids).to(device)

        model_params_group = [
            {
                "params": self.clothe_classifier.parameters(),
                "lr": config.OPTIMIZER.LEARNING_RATE,
                "weight_decay": 5e-4,
                "momentum": 0.9,
            }
        ]
        self.optimizer = optim.Adam(model_params_group)

        self.pid2clothes = torch.from_numpy(pid2clothes).to(device)

        self.criterion_ce = nn.CrossEntropyLoss().to(device)
        self.criterion_ce_ls = CrossEntropyLabelSmooth(num_classes=num_clothe_pids, epsilon=0.1, use_gpu=torch.cuda.is_available())
