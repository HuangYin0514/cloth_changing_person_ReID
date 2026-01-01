import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger, clothe_base):
    scheduler.step(epoch)
    reid_net.train()
    clothe_base.clothe_classifier.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)

        if config.MODEL.MODULE == "Lucky":
            B = img.size(0)
            total_loss = 0

            backbone_feat_map, global_feat, global_bn_feat = reid_net(img)

            if epoch > 25:
                clothe_cls_score = clothe_base.clothe_classifier(global_feat.detach())
                clothe_loss = clothe_base.criterion_ce_ls(clothe_cls_score, clotheid)
                meter.update({"clothe_loss": clothe_loss.item()})
                clothe_base.optimizer.zero_grad()
                clothe_loss.backward()
                clothe_base.optimizer.step()

            # Global
            global_cls_score = reid_net.global_classifier(global_bn_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            # global_tri_loss = criterion.tri(global_feat, pid)
            global_loss = global_id_loss
            meter.update({"global_loss": global_loss.item()})
            total_loss += global_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
