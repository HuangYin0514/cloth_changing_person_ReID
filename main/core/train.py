import torch
import torch.nn.functional as F
import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger):
    scheduler.step(epoch)  # for cc
    reid_net.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)

        if config.MODEL.MODULE == "Lucky":
            B = img.size(0)
            total_loss = 0

            backbone_feat_map = reid_net(img)

            global_feat = reid_net.global_pool(backbone_feat_map).view(B, reid_net.GLOBAL_DIM)
            global_bn_feat, global_cls_score = reid_net.global_classifier(global_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            global_tri_loss = criterion.tri(global_feat, pid)
            global_loss = global_id_loss + global_tri_loss
            total_loss += global_loss
            meter.update({"global_loss": global_loss.item()})

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
