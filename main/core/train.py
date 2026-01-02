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

            backbone_feat_map, global_feat, global_bn_feat, backbone_inside_feat_map = reid_net(img)

            # Global
            global_cls_score = reid_net.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            global_tri_loss = criterion.tri(global_feat, pid)
            global_loss = global_id_loss + global_tri_loss
            meter.update({"global_loss": global_loss.item()})
            total_loss += global_loss

            # Backbone inside
            backbone_l4_b0_feat_map = backbone_inside_feat_map["backbone_l4_b0_feat_map"]
            backbone_l4_b0_feat = reid_net.backbone_l4_b0_pool(backbone_l4_b0_feat_map).view(B, reid_net.GLOBAL_DIM)
            backbone_l4_b0_bn_feat = reid_net.backbone_l4_b0_neck(backbone_l4_b0_feat)
            backbone_l4_b0_part_list = reid_net.backbone_l4_b0_part_module(backbone_l4_b0_feat_map)
            backbone_l4_b0_g_p_feat = torch.cat([backbone_l4_b0_bn_feat, torch.cat(backbone_l4_b0_part_list, dim=1)], dim=1)
            backbone_l4_b0_cls_score = reid_net.backbone_l4_b0_classifier(backbone_l4_b0_g_p_feat)
            backbone_l4_b0_id_loss = criterion.ce_ls(backbone_l4_b0_cls_score, pid)
            backbone_l4_b0_tri_loss = criterion.tri(backbone_l4_b0_feat, pid)
            backbone_l4_b0_loss = backbone_l4_b0_id_loss + backbone_l4_b0_tri_loss
            meter.update({"backbone_l4_b0_loss": backbone_l4_b0_loss.item()})
            total_loss += backbone_l4_b0_loss

            backbone_l4_b1_feat_map = backbone_inside_feat_map["backbone_l4_b1_feat_map"]
            backbone_l4_b1_feat = reid_net.backbone_l4_b1_pool(backbone_l4_b1_feat_map).view(B, reid_net.GLOBAL_DIM)
            backbone_l4_b1_bn_feat = reid_net.backbone_l4_b1_neck(backbone_l4_b1_feat)
            backbone_l4_b1_part_list = reid_net.backbone_l4_b1_part_module(backbone_l4_b1_feat_map)
            backbone_l4_b1_g_p_feat = torch.cat([backbone_l4_b1_bn_feat, torch.cat(backbone_l4_b1_part_list, dim=1)], dim=1)
            backbone_l4_b1_cls_score = reid_net.backbone_l4_b1_classifier(backbone_l4_b1_g_p_feat)
            backbone_l4_b1_id_loss = criterion.ce_ls(backbone_l4_b1_cls_score, pid)
            backbone_l4_b1_tri_loss = criterion.tri(backbone_l4_b1_feat, pid)
            backbone_l4_b1_loss = backbone_l4_b1_id_loss + backbone_l4_b1_tri_loss
            meter.update({"backbone_l4_b1_loss": backbone_l4_b1_loss.item()})
            total_loss += backbone_l4_b1_loss

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
                unclothe_cam_tri_loss = criterion.tri(unclothe_cam_feat, pid)
                unclothe_cam_loss = unclothe_cam_id_loss + unclothe_cam_tri_loss
                meter.update({"unclothe_cam_loss": unclothe_cam_loss.item()})
                total_loss += unclothe_cam_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
