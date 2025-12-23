from bisect import bisect_right

import torch
import torch.nn as nn
import torch.optim as optim


class Build_Scheduler:
    def __init__(self, config, optimizer):
        self.name = "Scheduler"
        self.build(config, optimizer)

    def build(self, config, optimizer):
        self.scheduler = None

        if config.SCHEDULER.NAME == "WarmupMultiStepLR":
            self.scheduler = WarmupMultiStepLR(
                optimizer,
                [20, 40, 60],
                gamma=0.1,
                warmup_factor=0.01,
                warmup_iters=10,
                last_epoch=-1,
            )

        if config.SCHEDULER.NAME == "Adjust_Learning_Rate":
            self.scheduler = Adjust_Learning_Rate(config, optimizer)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs]

    def __repr__(self):
        return "{}(milestones={}, gamma={}, warmup_factor={}, warmup_iters={}, warmup_method={})".format(
            self.__class__.__name__,
            self.milestones,
            self.gamma,
            self.warmup_factor,
            self.warmup_iters,
            self.warmup_method,
        )


class Adjust_Learning_Rate:
    def __init__(self, config, optimizer):
        self.config = config
        self.optimizer = optimizer

    def step(self, epoch):
        CONFIG_LR = self.config.OPTIMIZER.LEARNING_RATE

        if epoch < 10:
            lr = CONFIG_LR * (epoch + 1) / 10
        elif epoch >= 10 and epoch < 20:
            lr = CONFIG_LR
        elif epoch >= 20 and epoch < 50:
            lr = CONFIG_LR * 0.1
        elif epoch >= 50:
            lr = CONFIG_LR * 0.01

        self.optimizer.param_groups[0]["lr"] = 0.1 * lr
        for i in range(len(self.optimizer.param_groups) - 1):
            self.optimizer.param_groups[i + 1]["lr"] = lr

        return lr
