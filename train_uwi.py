import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.ZNet import TransFuse_S, TransFuse_L, TransFuse_L_384
from utils.dataloader import get_uwi_loader, test_dataset, SUIM_test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os
from sklearn.metrics import f1_score

def dice_loss(pred, target):
    smooth = 1e-5  # 平滑项，避免除零错误
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    return dice_loss

def structure_loss(pred, mask):
    # weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    # wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    #
    # pred = torch.sigmoid(pred)
    # inter = ((pred * mask)*weit).sum(dim=(2, 3))
    # union = ((pred + mask)*weit).sum(dim=(2, 3))
    # wiou = 1 - (inter + 1)/(union - inter+1)
    # return (wbce + wiou).mean()
    # 假设 pred 的形状是 (batch_size, num_classes, height, width)
    # 假设 mask 的形状是 (batch_size, height, width)，包含类别索引 (0, 1, ..., num_classes-1)

    # 计算权重
    mask_one_hot = F.one_hot(mask, num_classes=8).permute(0, 3, 1, 2).float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_one_hot, kernel_size=31, stride=1, padding=15) - mask_one_hot)

    # 计算加权的交叉熵损失
    ce_loss = F.cross_entropy(pred, mask, reduction='none')
    ce_loss = (weit * ce_loss.unsqueeze(1)).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # 计算加权交并比 (IoU) 损失
    pred_softmax = F.softmax(pred, dim=1)
    inter = (pred_softmax * mask_one_hot * weit).sum(dim=(2, 3))
    union = (pred_softmax + mask_one_hot * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    #Dice loss
    dice = dice_loss(pred_softmax, mask_one_hot)

    # 返回总损失
    # return (ce_loss + wiou).mean()
    return (ce_loss + wiou + dice).mean()


# 计算交并比（IoU）
def intersection_over_union(pred, label, num_classes=6):
    # pred和label都是二维数组，表示预测和真实的像素类别
    # num_classes是一个整数，表示类别数目
    # 返回一个长度为num_classes的一维数组，表示每个类别的交并比
    assert pred.shape == label.shape
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        intersection = ((pred == i) & (label == i)).sum()
        union = ((pred == i) | (label == i)).sum()
        if union > 0:
            iou[i] = intersection / union
        else:
            iou[i] = np.nan
            # 表示该类别不存在于图像中
    return iou


# 计算平均交并比（MIoU）
def mean_intersection_over_union(pred, label, num_classes=6):
    # pred和label都是二维数组，表示预测和真实的像素类别
    # num_classes是一个整数，表示类别数目
    # 返回一个0~1之间的浮点数，表示平均交并比
    assert pred.shape == label.shape
    iou = intersection_over_union(pred, label, num_classes)
    # 使用 np.isnan 过滤掉为 NaN 的值
    valid_iou = iou[~np.isnan(iou)]
    # 计算平均 IoU
    miou = np.mean(valid_iou)
    return miou


def train(train_loader, model, optimizer, epoch, best_loss, best_mIOU, best_F1, best_acc):
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts, enhances, depths = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        enhances = Variable(enhances).cuda()
        depths = Variable(depths).cuda()

        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)

        loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

        # ---- backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))
            # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
            #       '[lateral-2: {:.4f}]'.
            #       format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record2.show()))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 1 == 0:
        meanloss, mIOU, F1, acc = test(model, opt.test_path)
        if mIOU > best_mIOU and F1 > best_F1: # and acc > best_acc:
            # print('new best loss: ', meanloss)
            print('new best mIOU: ', mIOU)
            print('new best F1: ', F1)
            print('new best acc: ', acc)
            # best_loss = meanloss
            best_mIOU = mIOU
            best_F1 = F1
            best_acc = acc
            torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth' % epoch)
    return best_loss, best_mIOU, best_F1, best_acc


def test(model, path):
    model.eval()
    mean_loss = []

    # for s in ['val', 'test']:
    for s in ['test']:
        image_root = '{}/uwi_data_{}.npy'.format(path, s)
        gt_root = '{}/uwi_mask_{}.npy'.format(path, s)
        gtrgb_root = '{}/uwi_maskrgb_test.npy'.format(opt.test_path)
        enhance_root = '{}/uwi_enhance_test.npy'.format(opt.train_path)
        depth_root = '{}/uwi_depth_test.npy'.format(opt.train_path)
        filenme_root = '{}/uwi_testname_test.npy'.format(opt.test_path)
        test_loader = SUIM_test_dataset(image_root,enhance_root, depth_root, gt_root, gtrgb_root, filenme_root)

        # dice_bank = []
        # iou_bank = []
        loss_bank = []
        acc_bank = []
        mIOU_bank = []
        F1_bank = []

        for i in range(test_loader.size):
            image, gt, gt_rgb, enhance, depth, _ = test_loader.load_data()
            image = image.cuda()
            enhance = enhance.cuda()
            depth = depth.cuda()

            with torch.no_grad():
                _, _, res = model(image)
                _, res_max = torch.max(res.squeeze(), dim=0)
            loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).cuda().to(torch.int64))

            acc = np.sum(res_max.cpu().numpy() == gt.astype(np.int64)) / (res_max.shape[0] * res_max.shape[1])
            mIOU = mean_intersection_over_union(res_max.to(torch.uint8), torch.tensor(gt).cuda(), 8)
            F1 = f1_score(res_max.to(torch.uint8).cpu().flatten(), torch.tensor(gt).cpu().flatten(), average='macro')
            # res = res.sigmoid().data.cpu().numpy().squeeze()
            # gt = 1*(gt>0.5)
            # res = 1*(res > 0.5)

            # dice = mean_dice_np(gt, res)
            # iou = mean_iou_np(gt, res)
            # acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

            loss_bank.append(loss.item())
            # dice_bank.append(dice)
            # iou_bank.append(iou)
            acc_bank.append(acc)
            mIOU_bank.append(mIOU)
            F1_bank.append(F1)

        print('{} Loss: {:.4f}, mIOU: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.
              format(s, np.mean(loss_bank), np.mean(mIOU_bank), np.mean(F1_bank), np.mean(acc_bank)))

        mean_loss.append(np.mean(loss_bank))

    return mean_loss[0], np.mean(mIOU_bank), np.mean(F1_bank), np.mean(acc_bank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/SUIM/uwi_enhance_name/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/SUIM/uwi_enhance_name/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S_ori13_2.2test')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()

    # ---- build models ----
    model = TransFuse_S(pretrained=True).cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))

    image_root = '{}/uwi_data_train.npy'.format(opt.train_path)
    gt_root = '{}/uwi_mask_train.npy'.format(opt.train_path)
    enhance_root = '{}/uwi_enhance_train.npy'.format(opt.train_path)
    depth_root = '{}/uwi_depth_train.npy'.format(opt.train_path)

    train_loader = get_uwi_loader(image_root,enhance_root, depth_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    best_loss = 1e5
    best_mIOU = 0
    best_F1 = 0
    best_acc = 0
    for epoch in range(1, opt.epoch + 1):
        best_loss, best_mIOU, best_F1, best_acc = train(train_loader, model, optimizer, epoch, best_loss, best_mIOU,
                                                        best_F1, best_acc)
