import argparse
import os
import warnings

import torch
import torch.utils.data as data
import util
from data import Data_Loder, IdentitySampler
from model import ReIDNet
from visualization import visualization

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

    ######################################################################
    # Device
    DEVICE = torch.device(config.TASK.DEVICE)

    ######################################################################
    # Data
    # Dataset返回数据格式：图像，身份标签，摄像头标签，路径
    data_loder = Data_Loder(config)

    ######################################################################
    # Model
    net = ReIDNet(config, data_loder.N_class).to(DEVICE)
    util.resume_model(net, config.MODEL.RESUME_EPOCH, path=os.path.join(config.SAVE.OUTPUT_PATH, "models/"))

    ########################################################
    # 可视化
    ########################################################
    sampler = IdentitySampler(
        data_loder.trainset.color_label,
        data_loder.trainset.thermal_label,
        data_loder.color_pos,
        data_loder.thermal_pos,
        config.DATALOADER.NUM_INSTANCES,
        config.DATALOADER.BATCHSIZE,
        config.MODEL.RESUME_EPOCH,
    )
    data_loder.trainset.cIndex = sampler.index1  # color index
    data_loder.trainset.tIndex = sampler.index2  # thermal index

    # dataloder
    loader_batch = config.DATALOADER.BATCHSIZE * config.DATALOADER.NUM_INSTANCES
    train_loader = data.DataLoader(
        data_loder.trainset,
        batch_size=loader_batch,
        sampler=sampler,
        num_workers=config.DATALOADER.NUM_WORKERS,
        drop_last=True,
    )
    query_loader = data_loder.query_loader
    gallery_loader = data_loder.gallery_loader
    visualization(config, net, data_loder, train_loader, query_loader, gallery_loader, DEVICE)


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)
    run(config)
