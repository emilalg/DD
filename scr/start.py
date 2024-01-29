"""
    This is a wrapper for running the different models.
    You can either modify its parameters or use default ones. If the dataset is not found in default path,
    you must provide the breast-density-prediction.zip, 
    which will be extracted to the default path.
    
    @author: soronen
"""

import os
import zipfile
import random
import sys
import train

# Default path to the dataset. Relative path from this file.
DATASET_DESTINATION = "../data"

# ratios for train, validation and test sets. Make sure are ≤ 1!
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

RNG_SEED = 42 # default value, leave empty for true random seed


def initialize_dataset(path_dataset_zip):
    """
    Initializes the dataset.
    If the dataset is not found in default path, you must provide the breast-density-prediction.zip,
    which will be extracted to the default path.

    @param path_dataset_zip: Path to the dataset zip file.
    @return: Path to the dataset.
    """
    path_dataset = os.path.join(os.path.dirname(__file__), DATASET_DESTINATION)
    if not os.path.exists(path_dataset):
        print("Dataset not found. Extracting...")
        with zipfile.ZipFile(path_dataset_zip, "r") as zip_ref:
            zip_ref.extractall(os.path.join(os.path.dirname(__file__), DATASET_DESTINATION))
        print("Extraction done. ")
    return path_dataset


def randomize_dataset_sets(
    path_dataset, train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO, test_ratio=TEST_RATIO
):
    # data is jpg images in train/train/[breast_masks, dense_masks, images] folders
    # create a random seed and use save it to variable, do nothing else for now
    if (RNG_SEED):
        seed = RNG_SEED
    else:
        seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)
    print("Seed was:", seed)
    # create a list of all breast_mask images
    breast_masks = os.listdir(os.path.join(path_dataset, "train/train/breast_masks"))
    print("breast_masks:", breast_masks)
    # shuffle the breast_masks
    rng.shuffle(breast_masks)
    # create dir data/[seed]/ with subdirs train valid and test. and inside those directories breast_mask, dense_mask and input_image
    print("Creating directories...")
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "train", "breast_mask"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "train", "dense_mask"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "train", "input_image"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "validation", "breast_mask"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "validation", "dense_mask"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "validation", "input_image"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "test", "breast_mask"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "test", "dense_mask"))
    os.makedirs(os.path.join(path_dataset, "data", str(seed), "test", "input_image"))
    
    # split the shuffled images into train, validation and test sets based on the ratios. using slicing
    print("Splitting breast_masks into train, validation and test sets...")
    train_breast_masks = breast_masks[:int(len(breast_masks) * train_ratio)]
    validation_breast_masks = breast_masks[
        int(len(breast_masks) * train_ratio) : int(len(breast_masks) * (train_ratio + validation_ratio))
    ]
    test_breast_masks = breast_masks[int(len(breast_masks) * (train_ratio + validation_ratio)) :]
    
    
    # copy the images from path/train/train/breast_masks to data/[seed]/train/
    print("Copying breast_masks to train set...")
    for breast_mask in train_breast_masks:
        os.rename(
            os.path.join(path_dataset, "train/train/breast_masks", breast_mask),
            os.path.join(path_dataset, "data", str(seed), "train", "breast_mask", breast_mask),
        )

if __name__ == "__main__":
    train.main()