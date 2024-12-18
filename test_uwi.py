import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.ZNet import TransFuse_S, TransFuse_L, TransFuse_L_384
from utils.dataloader import test_dataset, SUIM_test_dataset
import imageio
from sklearn.metrics import f1_score

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

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

def palette_to_rgb(out):
    # 定义颜色调色板，格式为(R, G, B)
    color_palette = [
        (0, 0, 0),  # BW
        (0, 0, 255),  # HD
        (0, 255, 255),  # WR
        (255, 0, 0),  # RO
        (255, 0, 255),  # RI
        (255, 255, 0),  # FV
        (0, 255, 0),  # PF
        (255, 255, 255),  # SR
    ]
    height, width = out.shape
    # 将调色板转换为张量
    color_palette_tensor = torch.tensor(color_palette, dtype=torch.uint8)
    # 使用索引从调色板中获取颜色
    rgb_image = color_palette_tensor[out.view(-1)].view(height, width, 3)

    # return rgb_image.permute(2, 0, 1)
    return rgb_image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='snapshots/TransFuse_S_ori13_2.2/TransFuse-109.pth')
    parser.add_argument('--test_path', type=str,
                        default='data/SUIM/uwi_enhance_name/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='result/ori13_2.2_1002/', help='path to save inference segmentation')

    opt = parser.parse_args()

    model = TransFuse_S().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating models: ', opt.ckpt_path)

    image_root = '{}/uwi_data_test.npy'.format(opt.test_path)
    gt_root = '{}/uwi_mask_test.npy'.format(opt.test_path)
    gtrgb_root = '{}/uwi_maskrgb_test.npy'.format(opt.test_path)
    enhance_root = '{}/uwi_enhance_test.npy'.format(opt.test_path)
    depth_root = '{}/uwi_depth_test.npy'.format(opt.test_path)
    filenme_root = '{}/uwi_testname_test.npy'.format(opt.test_path)
    test_loader = SUIM_test_dataset(image_root,enhance_root, depth_root, gt_root, gtrgb_root, filenme_root)

    dice_bank = []
    iou_bank = []
    acc_bank = []
    mIOU_bank = []
    F1_bank = []

    for i in range(test_loader.size):
        image, gt, gt_rgb, enhance, depth, filename = test_loader.load_data()
        # gt = 1*(gt>0.5)
        image = image.cuda()
        depth = depth.cuda()

        with torch.no_grad():
            _, _, res = model(image, depth)

        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = 1*(res > 0.5)
        _, res_max = torch.max(res.squeeze(), dim=0)
        res_rgb = palette_to_rgb(res_max.cpu())

        if opt.save_path is not None:
            # imageio.imwrite(opt.save_path+'/'+str(i)+'_pred.jpg', np.uint8(res*255))
            imageio.imwrite(opt.save_path + '/' + filename.split('.')[0] + '_gt.jpg', gt_rgb)
            imageio.imwrite(opt.save_path + '/' + filename.split('.')[0] + '_pred.jpg', res_rgb)


        # dice = mean_dice_np(gt.astype(np.int64), res_max.cpu().numpy())
        # iou = mean_iou_np(gt.astype(np.int64), res_max.cpu().numpy())
        acc = np.sum(res_max.cpu().numpy() == gt.astype(np.int64)) / (res_max.shape[0]*res_max.shape[1])

        mIOU = mean_intersection_over_union(res_max.to(torch.uint8), torch.tensor(gt).cuda(), 8)
        F1 = f1_score(res_max.to(torch.uint8).cpu().flatten(), torch.tensor(gt).cpu().flatten(), average='macro')

        acc_bank.append(acc)
        mIOU_bank.append(mIOU)
        F1_bank.append(F1)
        # acc_bank.append(acc)
        # dice_bank.append(dice)
        # iou_bank.append(iou)

    print('mIOU: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.
        format(np.mean(mIOU_bank), np.mean(F1_bank), np.mean(acc_bank)))
