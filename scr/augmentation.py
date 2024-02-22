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

def augment_and_save(image_path, mask1_path, mask2_path, save_dirs, augmentation_pipeline):
    """A function to apply the augmentation to the image and masks and save the augmented images and masks in the directories
    :param image_path: path to the image
    :param mask1_path: path to the first mask
    :param mask2_path: path to the second mask
    :param save_dirs: contains the paths to save the augmented images and masks
    :param augmentation_pipeline: the augmenation pipeline to apply to the image and masks located in the config file
    """
    # read the image and masks
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask1 = cv2.imread(mask1_path,0)
    mask2 = cv2.imread(mask2_path,0)

    # apply the same augmentation to the image and masks
    augmented = augmentation_pipeline(image=image, masks=[mask1, mask2])
    augmented_image = augmented['image']
    augmented_mask1, augmented_mask2 = augmented['masks']

    # generates a random filename
    random_filename = f"{uuid.uuid4().hex}.jpg"

    # save the augmented image and masks
    cv2.imwrite(os.path.join(save_dirs['images'], random_filename), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dirs['masks1'], random_filename), augmented_mask1)
    cv2.imwrite(os.path.join(save_dirs['masks2'], random_filename), augmented_mask2)

# use augmentation needs to be true to apply the augmentation
config.use_augmentation = True
# get the augmentation pipeline from config file
augmentation_pipeline = get_augmentations(config)

# directories for training data
train_images_dir = 'breast-density-prediction/train/train/images'
train_masks1_dir = 'breast-density-prediction/train/train/breast_masks'
train_masks2_dir = 'breast-density-prediction/train/train/dense_masks'

# Directories for saving augmented data
augmented_images_dir = 'breast-density-prediction-test/train/train/images'
augmented_masks1_dir = 'breast-density-prediction-test/train/train/breast_masks'
augmented_masks2_dir = 'breast-density-prediction-test/train/train/dense_masks'

# copies the original training data to the augmented directory before applying the augmentation
if not os.path.exists('breast-density-prediction-test/train'):
    shutil.copytree('breast-density-prediction/train', 'breast-density-prediction-test/train')

# Ensure the augmented directories exist
for dir_path in [augmented_images_dir, augmented_masks1_dir, augmented_masks2_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# copy the train.csv to the augmented directory if it doesn't exist. Hypertuner requires train.csv
if not os.path.exists('breast-density-prediction-test/train/train.csv'):
    shutil.copy('breast-density-prediction/train.csv', 'breast-density-prediction-test/train.csv')

# list of training images and masks
train_images = os.listdir(train_images_dir)
train_masks1 = os.listdir(train_masks1_dir)
train_masks2 = os.listdir(train_masks2_dir)

save_dirs = {
    'images': augmented_images_dir,
    'masks1': augmented_masks1_dir,
    'masks2': augmented_masks2_dir
}

# applies the augmentatio. FOr loop is used to apply the augmentation multiple times.
# Looping 3 times creates about 1000 images and masks. 10 times abbout 6000 images and masks
# looping 10 times takes about 10 minutes, and takes 6gb of storage. So choose wisely :)
for img_name, mask1_name, mask2_name in zip(train_images, train_masks1, train_masks2):
    for _ in range(10):
        image_path = os.path.join(train_images_dir, img_name)
        mask1_path = os.path.join(train_masks1_dir, mask1_name)
        mask2_path = os.path.join(train_masks2_dir, mask2_name)
        augment_and_save(image_path, mask1_path, mask2_path, save_dirs, augmentation_pipeline)
