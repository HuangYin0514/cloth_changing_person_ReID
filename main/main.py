import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import util
from criterion import Criterion
from data import Data_Loder, IdentitySampler
from eval_metrics import eval_regdb, eval_sysu
from model import ReIDNet
from optimizer import Optimizer
from scheduler import Scheduler
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
    logger(config)

    # ######################################################################
    # # Device
    # DEVICE = torch.device(config.TASK.DEVICE)

    # ######################################################################
    # # Data
    # data_loder = Data_Loder(config)

    # ######################################################################
    # # Model
    # net = ReIDNet(config, data_loder.N_class).to(DEVICE)

    # ######################################################################
    # # Criterion
    # criterion = Criterion(config)

    # ######################################################################
    # # Optimizer
    # optimizer = Optimizer(config, net).optimizer

    # ######################################################################
    # # Scheduler
    # scheduler = Scheduler(config, optimizer)

    # ######################################################################
    # # Training & Evaluation
    # print("==> Start Training...")
    # # 初始化最佳指标
    # best_epoch, best_mAP, best_rank1 = 0, 0, 0
    # for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
    #     #########
    #     # train
    #     #########
    #     scheduler.lr_scheduler.step(epoch)
    #     # TODO
    #     meter = None
    #     logger("Time: {}; Epoch: {}; {}".format(util.time_now(), epoch, meter.get_str()))
    #     wandb.log({"Lr": optimizer.param_groups[0]["lr"], **meter.get_dict()})

    #     #########
    #     # Test
    #     #########
    #     if epoch % config.TEST.EVAL_EPOCH == 0:
    #         # TODO
    #         mAP, CMC = None, None
    #         logger("Time: {}; Test on Dataset: {}, \nmAP: {} \nRank: {}".format(util.time_now(), config.DATASET.TRAIN_DATASET, mAP, CMC))
    #         wandb.log({"test_epoch": epoch, "mAP": mAP, "Rank1": CMC[0]})

    # logger("=" * 50)
    # logger("Best model is: epoch: {}, rank1: {}, mAP: {}".format(best_epoch, best_rank1, best_mAP))
    # logger("=" * 50)


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)

    # 初始化wandb
    wandb.init(
        entity="yinhuang-team-projects",
        project=config.TASK.PROJECT,
        name=config.TASK.NAME,
        notes=config.TASK.NOTES,
        tags=config.TASK.TAGS,
        config=config,
    )
    run(config)
    wandb.finish()
