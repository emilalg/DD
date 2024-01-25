# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:23:11 2020

@author: rajgudhe
"""

import os
import glob
from natsort import natsorted
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

class MammoDataset(Dataset):
    def __init__(self, path, dataset, split, augmentations=False):
        self.path = path
        self.split = split
        self.dataset = dataset
        self.augmentations = augmentations
        #self.augmentation_type = augmentation_type

        
        self.images = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'input_image/*')))
        self.masks = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'breast_mask/*')))
        self.contours = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'dense_mask/*')))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(),])

        self.aug_pipeline = A.Compose([A.ShiftScaleRotate(p = 0.5),
                                       A.HorizontalFlip(0.5),
                                       A.VerticalFlip(0.5),
                                       A.RandomBrightnessContrast(0.5),
                                       A.GaussNoise(p = 0.5),
                                       A.GaussianBlur(p=0.5),
                                       A.ElasticTransform(),],p=0.8)


        self.image = cv2.imread(self.images[index], 1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.mask = cv2.imread(self.masks[index], 0)
        self.contour = cv2.imread(self.contours[index], 0)

        if self.augmentations:
            torch.manual_seed(1990)
            masks = [self.mask, self.contour]
            #if self.augmentation_type == 'pixel':
            #    self.sample = self.pixel_augmentations(image = self.image, masks = masks)
            #elif self.augmentation_type == 'spatial':
            #    self.sample = self.spatial_augmentations(image=self.image, masks=masks)
            #elif self.augmentation_type == 'pixel_spatial':
            #    self.sample = self.pixel_spatial_augmentations(image=self.image, masks=masks)
            #elif self.augmentation_type == 'aug_pipeline':
            self.sample = self.aug_pipeline(image=self.image, masks=masks)

            return self.to_tensor(self.sample['image']), self.to_tensor(self.sample['masks'][0]), self.to_tensor(self.sample['masks'][1])
        else:
            return self.to_tensor(self.image), self.to_tensor(self.mask), self.to_tensor(self.contour)

class MammoEvaluation(Dataset):
    def __init__(self, path, dataset, split):
        self.path = path
        self.split = split
        self.dataset = dataset

        self.images = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'input_image/*')))
        self.b_mask = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'breast_mask/*')))
        self.d_mask = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'dense_mask/*')))
        #self.images = natsorted(glob.glob(os.path.join(self.path,  '*')))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(), ])

        self.image_org = cv2.imread(self.images[index], 1)
        self.image_org = cv2.cvtColor(self.image_org, cv2.COLOR_BGR2RGB)

        self.b_mask_org = cv2.imread(self.b_mask[index], 0)
        self.d_mask_org = cv2.imread(self.d_mask[index], 0)

        
        self.image_org = self.to_tensor(self.image_org)
        self.b_mask_org = self.to_tensor(self.b_mask_org)
        self.d_mask_org = self.to_tensor(self.d_mask_org)
        #print(self.image)
        #print(self.image.shape)
        
        return os.path.split(self.images[index])[-1], self.image_org, self.b_mask_org, self.d_mask_org

### 
#data = KUHDataset(path='data', dataset='DDSM_Dataset', split='train')
#print(len(data))

