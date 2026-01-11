import torch
import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, clothe_base):
    reid_net.train()
    clothe_base.clothe_classifier_net.train()
    meter = util.MultiItemAverageMeter()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)
        total_loss = 0

        if config.MODEL.MODULE == "Lucky":
            B, C, H, W = img.size()
            backbone_feat_map, global_feat, global_bn_feat = reid_net(img)

            # Global
            global_cls_score = reid_net.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            meter.update({"global_id_loss": global_id_loss.item()})
            total_loss += global_id_loss
            if epoch > 35:
                global_tri_loss = criterion.tri(global_feat, pid)
                meter.update({"global_tri_loss": global_tri_loss.item()})
                total_loss += global_tri_loss

            # 定位
            clothe_cls_score = clothe_base.clothe_classifier_net(backbone_feat_map.detach())
            clothe_loss = clothe_base.criterion_ce(clothe_cls_score, clotheid)
            meter.update({"clothe_loss": clothe_loss.item()})
            clothe_base.optimizer.zero_grad()
            clothe_loss.backward()
            clothe_base.optimizer.step()
            clothe_feat_map = reid_net.clothe_cam_position(backbone_feat_map, clotheid, clothe_base.clothe_classifier_net)

            # 去除衣服
            unclothe_cam_feat_map = backbone_feat_map - clothe_feat_map
            unclothe_cam_feat = reid_net.clothe_cam_pool(unclothe_cam_feat_map).view(B, reid_net.GLOBAL_DIM)
            unclothe_cam_feat_bn_feat = reid_net.clothe_cam_bn_neck(unclothe_cam_feat)
            unclothe_cam_cls_score = reid_net.clothe_cam_classifier(unclothe_cam_feat_bn_feat)
            unclothe_cam_id_loss = criterion.ce_ls(unclothe_cam_cls_score, pid)
            meter.update({"unclothe_cam_id_loss": unclothe_cam_id_loss.item()})
            total_loss += unclothe_cam_id_loss
            if epoch > 35:
                unclothe_cam_tri_loss = criterion.tri(unclothe_cam_feat, pid)
                meter.update({"unclothe_cam_tri_loss": unclothe_cam_tri_loss.item()})
                total_loss += unclothe_cam_tri_loss

            # 蒸馏
            propagation_loss = reid_net.propagation(student_logits=global_cls_score, teacher_logits=unclothe_cam_cls_score)
            meter.update({"propagation_loss": propagation_loss.item()})
            total_loss += 0.01 * propagation_loss / B

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    scheduler.step()
    return meter
