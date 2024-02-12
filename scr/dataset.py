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
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import albumentations as A
from collections import namedtuple
import csv

SplitRatios = namedtuple('SplitRatios', ['train', 'valid'])
split_ratios = SplitRatios(train=0.75, valid=0.25)

def save_files_for_evaluation (model_name, whole_dataset, train_set, val_set, test_set, generator=torch.Generator().manual_seed(42)):
    train_set_files = [os.path.basename(whole_dataset[i]) for i in train_set.indices]
    val_set_files = [os.path.basename(whole_dataset[i]) for i in val_set.indices]
    with open(os.path.join(os.path.dirname(__file__), f"../test_output/logs/{model_name}_splits.txt"), 'w') as f:
        f.write(f'Seed: {generator.seed()}\n')
        f.write('Train:\n')
        for filename in train_set_files: 
            f.write(filename + '\n') 
        f.write('Valid:\n')
        for filename in val_set_files:
            f.write(filename + '\n') 

def split_orderly(whole_dataset, split_ratios):
    train_set = torch.utils.data.Subset(whole_dataset, 
                                        range(int(len(whole_dataset) * split_ratios.train)))
    val_set = torch.utils.data.Subset(whole_dataset, 
                                      range(int(len(whole_dataset) * split_ratios.train),
                                            int(len(whole_dataset) * (split_ratios.train + split_ratios.valid))))
    return train_set, val_set


def get_dataset_splits(path, model_name, split_ratios=split_ratios, generator=torch.Generator().manual_seed(42)):
    # check if path is empty
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path {path} does not exist! the directory should have breast_masks, dense_masks and images -folders' )
    if (split_ratios.train + split_ratios.valid) > 1.0:
        raise ValueError(f'Split ratios should be less than or equal to 1. Current split ratios are {split_ratios}')
   
    whole_dataset = natsorted(glob.glob(os.path.join(path, 'images/*')))
    # split the file names into train, valid and test sets
    train_set, val_set = random_split(whole_dataset, [split_ratios.train, split_ratios.valid], generator=generator)
    
    # alternatively just pick them orderly
    # train_set, val_set, test_set = split_orderly(whole_dataset, split_ratios)
    
    # save the split to a file. maybe it will be used for evaluation
    save_files_for_evaluation(model_name, whole_dataset, train_set, val_set, generator)
    # convert train_set and val_set to list of filenames
    train_set = [os.path.basename(whole_dataset[i]) for i in train_set.indices]
    val_set = [os.path.basename(whole_dataset[i]) for i in val_set.indices]
    
    return (train_set, val_set)

def construct_val_set(model_name):
    with open(os.path.join(os.path.dirname(__file__), f"../test_output/logs/{model_name}_splits.txt"), 'r') as f:
        val_set = []
        line = f.readline()
        while line:
            if line == 'Valid:\n':
                while line:
                    if line.endswith('.jpg\n'):
                        val_set.append(line[:-1])
                    line = f.readline()
                break
            line = f.readline()
    return val_set


class MammoDataset(Dataset):
    """
    Mammodataset class it's used for loading and
    transforming mammography images and masks
    """
    def __init__(self, path, filenames, augmentations=False):
        """ used to initialize class with required parameters
        :param path: Path to the dataset
        :param dataset: Name of the dataset folder
        :param filenames: List of images from which the dataset will be created
        :param split: Type of datasetsplit what is used such as 'train', 'valid' or 'test'

        """
        self.path = path
        self.augmentations = augmentations
        #self.augmentation_type = augmentation_type

        
        self.images = natsorted([os.path.join(self.path, 'images', filename) for filename in filenames])
        self.masks = natsorted([os.path.join(self.path, 'breast_masks', filename) for filename in filenames])
        self.contours = natsorted([os.path.join(self.path, 'dense_masks', filename) for filename in filenames])
        #self.images = natsorted(glob.glob(os.path.join(self.path, 'input_images/*')))
        #self.masks = natsorted(glob.glob(os.path.join(self.path, 'breast_masks/*')))
        #self.contours = natsorted(glob.glob(os.path.join(self.path, 'images/*')))
        
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
        self.aug_pipeline = A.Compose([A.ShiftScaleRotate(),
                                       A.HorizontalFlip(),
                                       A.VerticalFlip(),
                                       A.GaussNoise(),
                                       A.GaussianBlur(),],p=0.5)

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
    def __init__(self, path, dataset, split, mode, ground_truths_path=None, mask_path=None, model_name=None):
        """ used to initialize class with required parameters
        :param path: Path to the dataset
        :param dataset: Name of the dataset folder
        :param split: Type of datasetsplit what is used such as 'train', 'valid' or 'test'
        :param mode: Mode of operation ('submission', 'testsubmission').
        :param mask_path: Path to the directory containing ground truth masks for the test set (optional)
        """
        self.path = path
        self.split = split
        self.dataset = dataset
        self.mode = mode
        self.ground_truths_path = ground_truths_path
        self.mask_path = mask_path if mask_path else path  # Use separate mask path if provided, otherwise use the same path as the images

        # Initialize image lists
        self.images = []
        self.b_mask = []
        self.d_mask = []

        if mode == 'submission':
            # In submission mode, load all test images
            self.images = natsorted(glob.glob(os.path.join(self.path, 'test', 'test', 'images', '*')))
        elif mode == 'testsubmission':
            # In testsubmission mode, load images as per ground_truths (train.csv)
            with open(self.ground_truths_path, 'r') as f:
                reader = csv.DictReader(f)
                ground_truth_filenames = [row['Filename'] for row in reader]
                val_set_filenames = construct_val_set(model_name)
                # print(val_set_filenames)
                filtered_filenames = [filename for filename in ground_truth_filenames if filename in val_set_filenames]

            # Load images based on filenames in the ground truth list
            for file_name in filtered_filenames:
                img_path = os.path.join(self.path, 'train', 'train', 'images', file_name)
                if os.path.exists(img_path):
                    self.images.append(img_path)
                    self.b_mask.append(os.path.join(self.path, 'train', 'train', 'breast_masks', file_name))
                    self.d_mask.append(os.path.join(self.path, 'train', 'train', 'dense_masks', file_name))
                else:
                    print(f"Warning: Image file {file_name} not found.")


        # Verify the number of images and masks loaded
        print(f"Total images loaded: {len(self.images)}")
        print(f"Total breast masks loaded: {len(self.b_mask)}")
        print(f"Total dense masks loaded: {len(self.d_mask)}")

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

        self.image_org = self.to_tensor(self.image_org)
        
        if self.mode == 'submission':
            return os.path.split(self.images[index])[-1], self.image_org
        else:
            self.b_mask_org = cv2.imread(self.b_mask[index], 0)
            self.d_mask_org = cv2.imread(self.d_mask[index], 0)

            self.b_mask_org = self.to_tensor(self.b_mask_org)
            self.d_mask_org = self.to_tensor(self.d_mask_org)
             
            return os.path.split(self.images[index])[-1], self.image_org, self.b_mask_org, self.d_mask_org

### 
#data = KUHDataset(path='data', dataset='DDSM_Dataset', split='train')
#print(len(data))

