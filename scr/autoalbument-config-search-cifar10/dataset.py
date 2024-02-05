import torch
import torch.utils.data
import os
from utils import load_env
from dataset import MammoDataset, get_dataset_splits
from torchvision import transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader
from natsort import natsorted
from torch.utils.data import Dataset

load_env()

DATA_PATH = os.getenv("DATA_PATH", "../breast-density-prediction/train/train")

class SearchDatasett(Dataset):

    def __init__(self, path, filenames, transform=None):
        self.path = path
        
        self.images = natsorted([os.path.join(self.path, 'images', filename) for filename in filenames])
        self.masks = natsorted([os.path.join(self.path, 'breast_masks', filename) for filename in filenames])
        self.contours = natsorted([os.path.join(self.path, 'dense_masks', filename) for filename in filenames])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(),])

        # Read image
        self.image = cv2.imread(self.images[index], 1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Read breast mask and dense mask
        self.mask = cv2.imread(self.masks[index], 0)
        self.contour = cv2.imread(self.contours[index], 0)

        return self.to_tensor(self.image), self.to_tensor(self.mask), self.to_tensor(self.contour)

# Get dataset splits
train_set, val_set, test_set = get_dataset_splits(DATA_PATH)

# Initialize SearchDataset with the path to your data and the list of filenames
search_dataset = SearchDataset(path=DATA_PATH, filenames=train_set, transform=None)

# Create DataLoader for training
train_dataloader = DataLoader(
    search_dataset,
    batch_size=5,
    shuffle=True,
    num_workers=5
)
