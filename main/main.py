import argparse
import os
import time
import warnings

import torch
import util
from build_clothes_base import Build_Clothe_BASE
from build_criterion import Build_Criterion
from build_optimizer import Build_Optimizer
from build_scheduler import Build_Scheduler
from core import test, train
from data import build_dataloader
from model import ReID_Net

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
    device = torch.device(config.TASK.DEVICE)
    logger("Device is:\t {}".format(device))
    logger("*" * 20)

    ######################################################################
    # Data
    end = time.time()
    dataset, train_loader, query_loader, gallery_loader = build_dataloader(config)
    logger("Data loading time:\t {:.3f}".format(time.time() - end))
    logger("*" * 20)

    ######################################################################
    # Model
    reid_net = ReID_Net(config, dataset.num_train_pids).to(device)

    ######################################################################
    # Criterion
    criterion = Build_Criterion(config, dataset.num_train_pids)

    # ######################################################################
    # Optimizer
    optimizer = Build_Optimizer(config, reid_net).optimizer

    ######################################################################
    # Scheduler
    scheduler = Build_Scheduler(config, optimizer).scheduler

    ######################################################################
    # Training & Evaluation
    # 初始化最佳指标
    best_epoch, best_mAP, best_rank1 = 0, 0, 0
    for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
        meter = train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch)
        logger("Epoch: {}; {}".format(epoch, meter.get_str()))
        wandb.log({"Lr": optimizer.param_groups[0]["lr"], **meter.get_dict()})

        if epoch % config.TEST.EVAL_EPOCH == 0 or epoch == config.OPTIMIZER.TOTAL_TRAIN_EPOCH - 1:
            logger("=====> Start Testing...")
            end = time.time()
            mAP, CMC = test(config, reid_net, query_loader, gallery_loader, device, logger)
            logger("reid time: {:.3f}s".format(time.time() - end))
            logger("Dataset: {}, \t mAP: {:.4f}%; \t R-1: {:.4f}%.".format(config.DATA.TRAIN_DATASET, mAP * 100, CMC[0] * 100))
            wandb.log({"test_epoch": epoch, "mAP": mAP, "Rank1": CMC[0]})

            # is_best_rank_flag = CMC[0] >= best_rank1
            is_best_map_flag = mAP >= best_mAP
            if is_best_map_flag:
                best_epoch = epoch
                best_rank1 = CMC[0]
                best_mAP = mAP
                wandb.log({"best_epoch": best_epoch, "best_rank1": best_rank1, "best_mAP": best_mAP})
                # if epoch > 40:
                #     util.save_model(model=reid_net, epoch=epoch, path_dir=os.path.join(config.SAVE.OUTPUT_PATH, "models/"))

    logger("=" * 50)
    logger("Best model is: epoch: {}, mAP: {:.4f}%, rank1: {:.4f}%.".format(best_epoch, best_mAP * 100, best_rank1 * 100))
    logger("=" * 50)


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
