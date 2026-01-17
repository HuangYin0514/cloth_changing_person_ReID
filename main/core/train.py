import torch
import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, clothe_base):
    scheduler.step(epoch)
    reid_net.train()
    clothe_base.clothe_classifier.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)

        if config.MODEL.MODULE == "Lucky":
            B, C, H, W = img.size()
            total_loss = 0

            backbone_feat_map, global_feat, global_bn_feat = reid_net(img)

            # ------------- 全局信息 -----------------------
            global_cls_score = reid_net.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            meter.update({"global_id_loss": global_id_loss.item()})
            total_loss += global_id_loss
            global_tri_loss = criterion.tri(global_feat, pid)
            meter.update({"global_tri_loss": global_tri_loss.item()})
            total_loss += global_tri_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
