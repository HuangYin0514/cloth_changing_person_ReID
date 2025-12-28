import util
from tqdm import tqdm


def train(config, reid_net, train_loader, criterion, optimizer, scheduler, device, epoch, logger):
    scheduler.step(epoch)
    reid_net.train()
    meter = util.MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(train_loader)):
        img, black_img, pid, camid, clotheid = data
        img, pid, camid, clotheid = img.to(device), pid.to(device), camid.to(device), clotheid.to(device)
        black_img = black_img.to(device)

        if config.MODEL.MODULE == "Lucky":
            B = img.size(0)
            total_loss = 0

            backbone_feat_map = reid_net(img)

            global_feat = reid_net.global_pool(backbone_feat_map).view(B, reid_net.GLOBAL_DIM)
            global_bn_feat, global_cls_score = reid_net.global_classifier(global_feat)
            global_id_loss = criterion.ce_ls(global_cls_score, pid)
            global_tri_loss = criterion.tri(global_feat, pid)
            global_loss = global_id_loss + global_tri_loss
            meter.update({"global_loss": global_loss.item()})
            total_loss += global_loss

            black_feat_map = reid_net(black_img)

            black_feat = reid_net.mask_pool(black_feat_map).view(B, reid_net.GLOBAL_DIM)
            black_bn_feat, black_cls_score = reid_net.mask_classifier(black_feat)
            black_id_loss = criterion.ce_ls(black_cls_score, pid)
            black_tri_loss = criterion.tri(black_feat, pid)
            black_loss = black_id_loss + black_tri_loss
            meter.update({"mask_loss": black_loss.item()})
            total_loss += black_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return meter
