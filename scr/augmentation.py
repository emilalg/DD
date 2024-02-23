"""
augmentation.py is used to apply the augmentation to the images and masks. It uses the albumentations library to apply the augmentation.
It takes the images and masks from the training data /breast-density-prediction/ and saves the augmented images and masks in the breast-density-prediction-test directory.
"""

import cv2
import os
import uuid
from albumentations import Compose, HorizontalFlip, Rotate
from utils import get_augmentations, Config
import shutil

config = Config()

def augment_and_save(image_path, breast_mask_path, dense_mask_path, save_dirs, augmentation_pipeline):
    """A function to apply the augmentation to the image and masks and save the augmented images and masks in the directories
    :param image_path: path to the image
    :param breast_mask_path: path to the first mask
    :param dense_mask_path: path to the second mask
    :param save_dirs: contains the paths to save the augmented images and masks
    :param augmentation_pipeline: the augmenation pipeline to apply to the image and masks located in the config file
    """
    # read the image and masks
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask1 = cv2.imread(breast_mask_path,0)
    mask2 = cv2.imread(dense_mask_path,0)

    # apply the same augmentation to the image and masks
    augmented = augmentation_pipeline(image=image, masks=[mask1, mask2])
    augmented_image = augmented['image']
    augmented_breast_mask, augmented_dense_mask = augmented['masks']

    # generates a random filename
    random_filename = f"{uuid.uuid4().hex}.jpg"

    # save the augmented image and masks
    cv2.imwrite(os.path.join(save_dirs['images'], random_filename), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dirs['breast_masks'], random_filename), augmented_breast_mask)
    cv2.imwrite(os.path.join(save_dirs['dense_masks'], random_filename), augmented_dense_mask)

# use augmentation needs to be true to apply the augmentation
config.use_augmentation = True
# get the augmentation pipeline from config file
augmentation_pipeline = get_augmentations(config)

# directories for training data
train_images_dir = 'breast-density-prediction/train/train/images'
train_breast_masks_dir = 'breast-density-prediction/train/train/breast_masks'
train_dense_masks_dir = 'breast-density-prediction/train/train/dense_masks'

# Directories for saving augmented data
augmented_images_dir = 'breast-density-prediction-test/train/train/images'
augmented_breast_masks_dir = 'breast-density-prediction-test/train/train/breast_masks'
augmented_dense_masks_dir = 'breast-density-prediction-test/train/train/dense_masks'

# copies the original training data to the augmented directory before applying the augmentation
if not os.path.exists('breast-density-prediction-test/train'):
    shutil.copytree('breast-density-prediction/train', 'breast-density-prediction-test/train')

# Ensure the augmented directories exist
for dir_path in [augmented_images_dir, augmented_breast_masks_dir, augmented_dense_masks_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# copy the train.csv to the augmented directory if it doesn't exist. Hypertuner requires train.csv
if not os.path.exists('breast-density-prediction-test/train/train.csv'):
    shutil.copy('breast-density-prediction/train.csv', 'breast-density-prediction-test/train.csv')

# list of training images and masks
train_images = os.listdir(train_images_dir)
train_breast_masks = os.listdir(train_breast_masks_dir)
train_dense_masks = os.listdir(train_dense_masks_dir)

save_dirs = {
    'images': augmented_images_dir,
    'breast_masks': augmented_breast_masks_dir,
    'dense_masks': augmented_dense_masks_dir
}

# applies the augmentatio. FOr loop is used to apply the augmentation multiple times.
# Looping 3 times creates about 1000 images and masks. 10 times abbout 6000 images and masks
# looping 10 times takes about 10 minutes, and takes 6gb of storage. So choose wisely :)
for img_name, breast_mask_name, dense_mask_name in zip(train_images, train_breast_masks, train_dense_masks):
    for _ in range(3):
        image_path = os.path.join(train_images_dir, img_name)
        breast_mask_path = os.path.join(train_breast_masks_dir, breast_mask_name)
        dense_mask_path = os.path.join(train_dense_masks_dir, dense_mask_name)
        augment_and_save(image_path, breast_mask_path, dense_mask_path, save_dirs, augmentation_pipeline)
