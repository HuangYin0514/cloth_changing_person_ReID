import argparse
import os
import time
import warnings

import torch
import torch.utils.data as data
import util
from core import visualization
from data import build_dataloader
from model import ReID_Net

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

    ######################################################################
    # Data
    end = time.time()
    dataset, train_loader, query_loader, gallery_loader = build_dataloader(config)
    logger("Data loading time:\t {:.3f}".format(time.time() - end))

    ######################################################################
    # Model
    reid_net = ReID_Net(config, dataset.num_train_pids).to(device)
    util.resume_model(reid_net, config.TEST.RESUME_EPOCH, path=os.path.join(config.SAVE.OUTPUT_PATH, "models/"))

    ########################################################
    # 可视化
    ########################################################
    visualization(config, reid_net, train_loader, query_loader, gallery_loader, logger, device)


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)
    run(config)
