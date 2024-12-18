import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        
        image = self.images[index]
        gt = self.gts[index]
        # gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        # gt = self.gt_transform(transformed['mask'])
        gt = transformed['mask']
        gt = torch.from_numpy(gt).long()
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SkinDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, gtrgb_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.gts_rgb = np.load(gtrgb_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt_rgb = self.gts_rgb[self.index]
        # gt = gt/255.0
        self.index += 1

        return image, gt, gt_rgb

class SUIM_test_dataset:
    def __init__(self, image_root, enhance_root, depth_root, gt_root, gtrgb_root, filename_root):
        self.images = np.load(image_root)
        self.enhances = np.load(enhance_root)
        self.depths = np.load(depth_root)
        self.gts = np.load(gt_root)
        self.gts_rgb = np.load(gtrgb_root)
        self.filename = np.load(filename_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        enhance = self.enhances[self.index]
        enhance = self.transform(enhance).unsqueeze(0)
        depth = self.depths[self.index]
        depth = torch.from_numpy(depth / 255.0).unsqueeze(0).unsqueeze(0)
        gt = self.gts[self.index]
        gt_rgb = self.gts_rgb[self.index]
        # gt = gt/255.0
        filename = self.filename[self.index]
        self.index += 1

        # transformed = self.transform(image=image, mask=gt, image1=enhance, mask1=depth)
        # image = self.img_transform(transformed['image'])
        # depth = transformed['mask1']
        # depth = self.gt_transform(transformed['mask1'])
        # depth = self.depth_transform(transformed['mask1'])
        # gt = self.gt_transform(transformed['mask'])
        # gt = transformed['mask']
        # gt = torch.from_numpy(gt).long()

        return image, gt, gt_rgb, enhance, depth, filename

class SUIMwithLA_test_dataset:
    def __init__(self, image_root, la_root, gt_root, gtrgb_root, filename_root):
        self.images = np.load(image_root)
        self.las = np.load(la_root)
        self.gts = np.load(gt_root)
        self.gts_rgb = np.load(gtrgb_root)
        self.filename = np.load(filename_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        la = self.las[self.index]
        la = self.transform(la).unsqueeze(0)
        # depth = self.depths[self.index]
        # depth = torch.from_numpy(depth / 255.0).unsqueeze(0).unsqueeze(0)
        gt = self.gts[self.index]
        gt_rgb = self.gts_rgb[self.index]
        # gt = gt/255.0
        filename = self.filename[self.index]
        self.index += 1

        # transformed = self.transform(image=image, mask=gt, image1=enhance, mask1=depth)
        # image = self.img_transform(transformed['image'])
        # depth = transformed['mask1']
        # depth = self.gt_transform(transformed['mask1'])
        # depth = self.depth_transform(transformed['mask1'])
        # gt = self.gt_transform(transformed['mask'])
        # gt = transformed['mask']
        # gt = torch.from_numpy(gt).long()

        return image, gt, gt_rgb, la, filename

class SUIMDataset(data.Dataset):
    """
    dataloader for underwater semantic segmentation tasks
    """

    def __init__(self, image_root, enhance_root, depth_root, gt_root):
        self.images = np.load(image_root)
        self.enhances = np.load(enhance_root)
        self.depths = np.load(depth_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ],
            additional_targets={
                'image1': 'image',
                'mask1': 'mask'
            }
        )

    def __getitem__(self, index):
        image = self.images[index]
        enhance = self.enhances[index]
        depth = self.depths[index]
        depth = depth/255.0
        gt = self.gts[index]
        # gt = gt/255.0

        transformed = self.transform(image=image, mask=gt, image1=enhance, mask1=depth)
        image = self.img_transform(transformed['image'])
        enhance = self.img_transform(transformed['image1'])
        # depth = transformed['mask1']
        depth = self.gt_transform(transformed['mask1'])
        # depth = self.depth_transform(transformed['mask1'])
        # gt = self.gt_transform(transformed['mask'])
        gt = transformed['mask']
        gt = torch.from_numpy(gt).long()
        return image, gt, enhance, depth

    def __len__(self):
        return self.size


def get_uwi_loader(image_root, enhance_root, depth_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = SUIMDataset(image_root, enhance_root, depth_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class SUIMwithLADataset(data.Dataset):
    """
    dataloader for underwater semantic segmentation tasks
    """

    def __init__(self, image_root, la_root, gt_root):
        self.images = np.load(image_root)
        self.las = np.load(la_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ],
            additional_targets={
                'image1': 'image'
            }
        )

    def __getitem__(self, index):
        image = self.images[index]
        la = self.las[index]
        gt = self.gts[index]
        # gt = gt/255.0

        transformed = self.transform(image=image, mask=gt, image1=la)
        image = self.img_transform(transformed['image'])
        la = self.img_transform(transformed['image1'])
        # depth = transformed['mask1']
        # depth = self.gt_transform(transformed['mask1'])
        # depth = self.depth_transform(transformed['mask1'])
        # gt = self.gt_transform(transformed['mask'])
        gt = transformed['mask']
        gt = torch.from_numpy(gt).long()
        return image, gt, la

    def __len__(self):
        return self.size

def get_uwiwithLA_loader(image_root, la_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = SUIMwithLADataset(image_root, la_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class OUCDataset(data.Dataset):
    """
    dataloader for underwater semantic segmentation tasks
    """

    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ],
            additional_targets={
                'image1': 'image',
                'mask1': 'mask'
            }
        )

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        # gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        # enhance = self.img_transform(transformed['image1'])
        # depth = transformed['mask1']
        # depth = self.gt_transform(transformed['mask1'])
        # depth = self.depth_transform(transformed['mask1'])
        # gt = self.gt_transform(transformed['mask'])
        gt = transformed['mask']
        gt = torch.from_numpy(gt).long()
        return image, gt

    def __len__(self):
        return self.size

def get_OUC_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = OUCDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class OUC_test_dataset:
    def __init__(self, image_root, gt_root, gtrgb_root, filename_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.gts_rgb = np.load(gtrgb_root)
        self.filename = np.load(filename_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt_rgb = self.gts_rgb[self.index]
        # gt = gt/255.0
        filename = self.filename[self.index]
        self.index += 1
        return image, gt, gt_rgb, filename

class caveSeg_test_dataset:
    def __init__(self, image_root, gt_root, gtrgb_root, filename_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.gts_rgb = np.load(gtrgb_root)
        self.filename = np.load(filename_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt_rgb = self.gts_rgb[self.index]
        # gt = gt/255.0
        filename = self.filename[self.index]
        self.index += 1

        return image, gt, gt_rgb, filename


class CaveSegDataset(data.Dataset):
    """
    dataloader for underwater semantic segmentation tasks
    """

    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        # gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        # gt = self.gt_transform(transformed['mask'])
        gt = transformed['mask']
        gt = torch.from_numpy(gt).long()
        return image, gt

    def __len__(self):
        return self.size


def get_caveSeg_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CaveSegDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class UWater_test_dataset:
    def __init__(self, image_root, filename_root):
        self.images = np.load(image_root)
        self.filename = np.load(filename_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        # enhance = self.enhances[self.index]
        # enhance = self.transform(enhance).unsqueeze(0)
        # depth = self.depths[self.index]
        # depth = torch.from_numpy(depth / 255.0).unsqueeze(0).unsqueeze(0)
        # gt = self.gts[self.index]
        # gt_rgb = self.gts_rgb[self.index]
        # gt = gt/255.0
        filename = self.filename[self.index]
        self.index += 1

        # transformed = self.transform(image=image, mask=gt, image1=enhance, mask1=depth)
        # image = self.img_transform(transformed['image'])
        # depth = transformed['mask1']
        # depth = self.gt_transform(transformed['mask1'])
        # depth = self.depth_transform(transformed['mask1'])
        # gt = self.gt_transform(transformed['mask'])
        # gt = transformed['mask']
        # gt = torch.from_numpy(gt).long()

        return image, filename

if __name__ == '__main__':
    path = 'data/'
    tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        plt.savefig('vis/'+str(i)+".jpg")
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')
