import torch
import util
from torchvision import transforms as T
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger):
    scheduler.step(epoch)
    reid_net.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, mask_img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)
        mask_img = mask_img.to(device)

        if config.MODEL.MODULE == "Lucky":
            B = img.size(0)
            total_loss = 0

            backbone_feat_map = reid_net(img)

            # ------------- Global -----------------------
            global_feat = reid_net.global_pool(backbone_feat_map).view(B, reid_net.GLOBAL_DIM)
            global_bn_feat, global_cls_score = reid_net.global_classifier(global_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            global_tri_loss = criterion.tri(global_feat, pid)
            global_loss = global_id_loss + global_tri_loss
            meter.update({"global_loss": global_loss.item()})
            total_loss += global_loss

            # ------------- Mask Attention -----------------------
            mask_att_feat_map = reid_net.mask_attention(backbone_feat_map)
            mask_feat_map = mask_att_feat_map[:, 0, :, :].unsqueeze(dim=1) * backbone_feat_map
            mask_feat = reid_net.mask_pool(mask_feat_map).view(B, reid_net.GLOBAL_DIM)
            mask_bn_feat, mask_cls_score = reid_net.mask_classifier(mask_feat)
            match_mask_loss = criterion.match_mask(mask_att_feat_map, mask_img)
            mask_id_loss = criterion.ce_ls(mask_cls_score, pid)
            mask_tri_loss = criterion.tri(mask_feat, pid)
            mask_loss = mask_id_loss + mask_tri_loss + 0.0 * match_mask_loss
            meter.update({"mask_loss": mask_loss.item()})
            total_loss += mask_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
