import torch
import torch.nn as nn
import torch.optim as optim
from loss import CosFaceLoss
from model import NormalizedClassifier


class Build_Clothe_BASE:

    def __init__(self, config, *args, **kwargs):
        super(Build_Clothe_BASE, self).__init__()
        self.build(config, *args, **kwargs)

    def build(self, config, global_dim, num_clothe_pids, pid2clothes, device):
        self.clothe_classifier = NormalizedClassifier(global_dim, num_clothe_pids).to(device)

        model_params_group = [{"params": self.clothe_classifier.parameters(), "lr": config.OPTIMIZER.LEARNING_RATE, "weight_decay": 5e-4, "momentum": 0.9}]
        self.optimizer = optim.Adam(model_params_group)

        self.pid2clothes = torch.from_numpy(pid2clothes).to(device)

        self.criterion_ce = nn.CrossEntropyLoss().to(device)
        self.criterion_cfl = CosFaceLoss(scale=16.0, margin=0).to(device)
        self.criterion_adv = ClothesBasedAdversarialLoss(device=device)


class ClothesBasedAdversarialLoss(nn.Module):
    """Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """

    def __init__(self, scale=16, epsilon=0.1, device=None):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with
                the same identity as the anchor sample are defined as positive clothes classes and their mask
                values are 1. The clothes classes with different identities from the anchor sample are defined
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).to(self.device)

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (-mask * log_prob).sum(1).mean()

        return loss
