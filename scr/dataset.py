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
    """
    Mammodataset class it's used for loading and
    transforming mammography images and masks
    """
    def __init__(self, path, dataset, split, augmentations=False):
        """ used to initialize class with required parameters
        :param path: Path to the dataset
        :param dataset: Name of the dataset folder
        :param split: Type of datasetsplit what is used such as 'train', 'valid' or 'test'

        """
        self.path = path
        self.split = split
        self.dataset = dataset
        self.augmentations = augmentations
        #self.augmentation_type = augmentation_type

        
        self.images = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'input_image/*')))
        self.masks = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'breast_mask/*')))
        self.contours = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'dense_mask/*')))
        
    def __len__(self):
        """
        :returns: number of items in the dataset
        """
        return len(self.images)
    
    def __getitem__(self, index):
        """ A function to retrieve single image and corresponding masks
        :param index: Index of the data
        """

        #Converts images into suitable format. First convert to PIL image. Then resize to 256x256. Lastly converts PIL image to tensor and scales the values accordingly.
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
        self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(),])

        # Uses Albumentations library that is a computer vision tool that can be used to create different variations of the same picture.
        # Slightly modifes the images to increase the size of training data.
        # Example: A.ShiftScaleRotate(p = 0.5) randomly apploes translate,scale and rotate to the input. 0.5 is the probability of applying the transform.
        #https://albumentations.ai/
        self.aug_pipeline = A.Compose([A.ShiftScaleRotate(p = 0.5),
                                       A.HorizontalFlip(0.5),
                                       A.VerticalFlip(0.5),
                                       A.RandomBrightnessContrast(0.5),
                                       A.GaussNoise(p = 0.5),
                                       A.GaussianBlur(p=0.5),
                                       A.ElasticTransform(),],p=0.8)

        # Read image
        self.image = cv2.imread(self.images[index], 1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Read breast mask and dense mask
        self.mask = cv2.imread(self.masks[index], 0)
        self.contour = cv2.imread(self.contours[index], 0)

        # if augmentation is true (defaulted to false) applies the modification of images to increate the size of training data (self.aug_pipeline)
        if self.augmentations:
            # Seed for generating random numbers
            torch.manual_seed(1990)
            # List containing dense and breast masks
            masks = [self.mask, self.contour]
            #if self.augmentation_type == 'pixel':
            #    self.sample = self.pixel_augmentations(image = self.image, masks = masks)
            #elif self.augmentation_type == 'spatial':
            #    self.sample = self.spatial_augmentations(image=self.image, masks=masks)
            #elif self.augmentation_type == 'pixel_spatial':
            #    self.sample = self.pixel_spatial_augmentations(image=self.image, masks=masks)
            #elif self.augmentation_type == 'aug_pipeline':

            # Store augmented image
            self.sample = self.aug_pipeline(image=self.image, masks=masks)
            # Return augmented image if augmentation
            return self.to_tensor(self.sample['image']), self.to_tensor(self.sample['masks'][0]), self.to_tensor(self.sample['masks'][1])
        else:
            # Return non augmented if no augmentation
            return self.to_tensor(self.image), self.to_tensor(self.mask), self.to_tensor(self.contour)

class MammoEvaluation(Dataset):
    def __init__(self, path, dataset, split):
        """ used to initialize class with required parameters
        :param path: Path to the dataset
        :param dataset: Name of the dataset folder
        :param split: Type of datasetsplit what is used such as 'train', 'valid' or 'test'

        """
        self.path = path
        self.split = split
        self.dataset = dataset

        self.images = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'input_image/*')))
        self.b_mask = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'breast_mask/*')))
        self.d_mask = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'dense_mask/*')))
        #self.images = natsorted(glob.glob(os.path.join(self.path,  '*')))


    def __len__(self):
        """
        :returns: number of items in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """ A function to retrieve the image and corresponding breast and dense mask
        :param index: Index of the data
        """
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

