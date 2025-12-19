import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import util
from build_criterion import Build_Criterion
from build_optimizer import Build_Optimizer
from build_scheduler import Build_Scheduler
from core import train
from data import build_dataloader
from model import ReID_Net
from tqdm import tqdm

import wandb

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="main/cfg/test.yml", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def run(config):
    ######################################################################
    # Logger
    logger = util.Logger(path_dir=os.path.join(config.SAVE.OUTPUT_PATH, "logs/"), name="logger.log")
    logger("Config:\t" + "*" * 20)
    logger(config)
    logger("*" * 20)

    ######################################################################
    # Device
    DEVICE = torch.device(config.TASK.DEVICE)
    logger("Device is:\t {}".format(DEVICE))

    ######################################################################
    # Data
    end = time.time()
    dataset, train_loader, query_loader, gallery_loader = build_dataloader(config)
    logger("Data loading time:\t {:.3f}".format(time.time() - end))

    ######################################################################
    # Model
    reid_net = ReID_Net(config, dataset.num_train_pids).to(DEVICE)
    # logger("Model:\n {}".format(reid_net))

    ######################################################################
    # Criterion
    criterion = Build_Criterion(config)
    logger("Criterion:\t {}".format(criterion))

    # ######################################################################
    # Optimizer
    optimizer = Build_Optimizer(config, reid_net).optimizer
    logger("Optimizer:\t {}".format(optimizer))

    ######################################################################
    # Scheduler
    scheduler = Build_Scheduler(config, optimizer).scheduler
    logger("Scheduler:\t {}".format(scheduler))

    ######################################################################
    # Training & Evaluation
    logger("=====> Start Training...")
    # 初始化最佳指标
    best_epoch, best_mAP, best_rank1 = 0, 0, 0
    for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
        meter = train(config, logger, epoch, reid_net, train_loader, criterion, optimizer, scheduler)
        logger("Time: {}; Epoch: {}; {}".format(util.time_now(), epoch, meter.get_str()))
        # wandb.log({"Lr": optimizer.param_groups[0]["lr"], **meter.get_dict()})

    #     # #########
    #     # # Test
    #     # #########
    #     # if epoch % config.TEST.EVAL_EPOCH == 0:
    #     #     # TODO
    #     #     mAP, CMC = None, None
    #     #     logger("Time: {}; Test on Dataset: {}, \nmAP: {} \nRank: {}".format(util.time_now(), config.DATASET.TRAIN_DATASET, mAP, CMC))
    #     #     wandb.log({"test_epoch": epoch, "mAP": mAP, "Rank1": CMC[0]})

    # logger("=" * 50)
    # logger("Best model is: epoch: {}, rank1: {}, mAP: {}".format(best_epoch, best_rank1, best_mAP))
    # logger("=" * 50)


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)

    # 初始化wandb
    # wandb.init(
    #     entity="yinhuang-team-projects",
    #     project=config.TASK.PROJECT,
    #     name=config.TASK.NAME,
    #     notes=config.TASK.NOTES,
    #     tags=config.TASK.TAGS,
    #     config=config,
    # )
    run(config)
    # wandb.finish()
