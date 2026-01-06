import torch
import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger, clothe_base):
    scheduler.step(epoch)
    reid_net.train()
    clothe_base.clothe_classifier_net.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)

        if config.MODEL.MODULE == "Lucky":
            B, C, H, W = img.size()
            total_loss = 0
            start_epoch = 25

            backbone_feat_map, global_feat, global_bn_feat = reid_net(img)

            # Global
            global_cls_score = reid_net.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            meter.update({"global_id_loss": global_id_loss.item()})
            total_loss += global_id_loss
            global_tri_loss = criterion.tri(global_feat, pid)
            meter.update({"global_tri_loss": global_tri_loss.item()})
            total_loss += global_tri_loss

            # # Part
            # part_feat_list, part_cls_score_list = reid_net.part_module(backbone_feat_map, global_bn_feat)
            # num_part = reid_net.part_module.num_part // 4
            # for i in range(num_part):
            #     part_id_loss = criterion.ce_ls(part_cls_score_list[i], pid)
            #     meter.update({"part_id_loss": part_id_loss.item()})
            #     total_loss += 1 / num_part * part_id_loss
            #     part_tri_loss = criterion.tri(part_feat_list[i], pid)
            #     meter.update({"part_tri_loss": part_tri_loss.item()})
            #     total_loss += 1 / num_part * part_tri_loss

            if epoch > -1:
                clothe_cls_score = clothe_base.clothe_classifier_net(backbone_feat_map.detach())
                clothe_loss = clothe_base.criterion_ce(clothe_cls_score, clotheid)
                meter.update({"clothe_loss": clothe_loss.item()})
                clothe_base.optimizer.zero_grad()
                clothe_loss.backward()
                clothe_base.optimizer.step()

                clothe_feat_map = reid_net.clothe_cam_position(backbone_feat_map, clotheid, clothe_base.clothe_classifier_net)
                unclothe_cam_feat_map = backbone_feat_map - clothe_feat_map
                unclothe_cam_feat = reid_net.clothe_cam_pool(unclothe_cam_feat_map).view(B, reid_net.GLOBAL_DIM)
                unclothe_cam_feat_bn_feat = reid_net.clothe_cam_bn_neck(unclothe_cam_feat)
                unclothe_cam_cls_score = reid_net.clothe_cam_classifier(unclothe_cam_feat_bn_feat)
                unclothe_cam_id_loss = criterion.ce_ls(unclothe_cam_cls_score, pid)
                meter.update({"unclothe_cam_id_loss": unclothe_cam_id_loss.item()})
                total_loss += unclothe_cam_id_loss
                unclothe_cam_tri_loss = criterion.tri(unclothe_cam_feat, pid)
                meter.update({"unclothe_cam_tri_loss": unclothe_cam_tri_loss.item()})
                total_loss += unclothe_cam_tri_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
