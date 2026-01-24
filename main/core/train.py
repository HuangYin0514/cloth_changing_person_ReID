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
        if config.DATA.TRAIN_DATASET == "ltcc":
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

                # ------------- 衣服分类器 -----------------------
                clothe_cls_score = clothe_base.clothe_classifier(backbone_feat_map.detach())
                clothe_loss = clothe_base.criterion_ce(clothe_cls_score, clotheid)
                meter.update({"clothe_loss": clothe_loss.item()})
                clothe_base.optimizer.zero_grad()
                clothe_loss.backward()
                clothe_base.optimizer.step()

                # ------------- 衣服定位 -----------------------
                clothe_feat_map = reid_net.clothe_position(backbone_feat_map, clotheid, clothe_base.clothe_classifier)

                # ------------- 衣服校准 -----------------------
                clothe_feat_map = reid_net.clothe_correction(backbone_feat_map, clothe_feat_map)
                correction_clothe_cls_score = clothe_base.clothe_classifier(clothe_feat_map)
                correction_clothe_loss = criterion.ce_ls(correction_clothe_cls_score, clotheid)
                meter.update({"correction_clothe_loss": correction_clothe_loss.item()})
                total_loss += correction_clothe_loss

                # ------------- 去除衣服 -----------------------
                unclothe_feat_map = torch.clamp(backbone_feat_map - clothe_feat_map, min=0)

                # ------------- 非衣服区域判别 -----------------------
                unclothe_feat = reid_net.unclothe_pool(unclothe_feat_map).view(B, reid_net.GLOBAL_DIM)
                unclothe_feat_bn_feat = reid_net.unclothe_bn_neck(unclothe_feat)
                unclothe_cls_score = reid_net.unclothe_classifier(unclothe_feat_bn_feat)
                unclothe_id_loss = criterion.ce_ls(unclothe_cls_score, pid)
                meter.update({"unclothe_id_loss": unclothe_id_loss.item()})
                total_loss += unclothe_id_loss
                unclothe_tri_loss = criterion.tri(unclothe_feat, pid)
                meter.update({"unclothe_tri_loss": unclothe_tri_loss.item()})
                total_loss += unclothe_tri_loss

                # ------------- 蒸馏 -----------------------
                propagation_loss = 0.1 * criterion.propagation(student_logits=global_cls_score, teacher_logits=unclothe_cls_score)
                meter.update({"propagation_loss": propagation_loss.item()})
                total_loss += propagation_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        if config.DATA.TRAIN_DATASET == "prcc":
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

                if epoch > 25:
                    # ------------- 衣服分类器 -----------------------
                    clothe_cls_score = clothe_base.clothe_classifier(backbone_feat_map.detach())
                    clothe_loss = clothe_base.criterion_ce(clothe_cls_score, clotheid)
                    meter.update({"clothe_loss": clothe_loss.item()})
                    clothe_base.optimizer.zero_grad()
                    clothe_loss.backward()
                    clothe_base.optimizer.step()

                    # ------------- 衣服定位 -----------------------
                    clothe_feat_map = reid_net.clothe_position(backbone_feat_map, clotheid, clothe_base.clothe_classifier)

                    # ------------- 衣服校准 -----------------------
                    clothe_feat_map = reid_net.clothe_correction(backbone_feat_map, clothe_feat_map)
                    correction_clothe_cls_score = clothe_base.clothe_classifier(clothe_feat_map)
                    correction_clothe_loss = criterion.ce_ls(correction_clothe_cls_score, clotheid)
                    meter.update({"correction_clothe_loss": correction_clothe_loss.item()})
                    total_loss += correction_clothe_loss

                    # ------------- 去除衣服 -----------------------
                    unclothe_feat_map = torch.clamp(backbone_feat_map - clothe_feat_map, min=0)

                    # ------------- 非衣服区域判别 -----------------------
                    unclothe_feat = reid_net.unclothe_pool(unclothe_feat_map).view(B, reid_net.GLOBAL_DIM)
                    unclothe_feat_bn_feat = reid_net.unclothe_bn_neck(unclothe_feat)
                    unclothe_cls_score = reid_net.unclothe_classifier(unclothe_feat_bn_feat)
                    unclothe_id_loss = criterion.ce_ls(unclothe_cls_score, pid)
                    meter.update({"unclothe_id_loss": unclothe_id_loss.item()})
                    total_loss += unclothe_id_loss
                    unclothe_tri_loss = criterion.tri(unclothe_feat, pid)
                    meter.update({"unclothe_tri_loss": unclothe_tri_loss.item()})
                    total_loss += unclothe_tri_loss

                    # ------------- 蒸馏 -----------------------
                    propagation_loss = 0.1 * criterion.propagation(student_logits=global_cls_score, teacher_logits=unclothe_cls_score)
                    meter.update({"propagation_loss": propagation_loss.item()})
                    total_loss += propagation_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
