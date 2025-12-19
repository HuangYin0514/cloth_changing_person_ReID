import torch
import torch.nn.functional as F
import util
from tqdm import tqdm


def train(config, logger, epoch, reid_net, train_loader, criterion, optimizer, scheduler):
    scheduler.step(epoch)
    reid_net.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clothes_id = data
        meter.update({"local_loss": 0})
    return meter
