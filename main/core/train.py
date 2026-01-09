import torch
import util
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value.

    Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch):
    reid_net.train()
    meter = util.MultiItemAverageMeter()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)

        feat_list, y_list = reid_net(img)

        loss = 0
        for index, y in enumerate(y_list):
            id_loss = criterion.ce_ls(y, pid)
            meter.update({"id_loss_{}".format(index): id_loss.item()})
            loss += id_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    return meter
