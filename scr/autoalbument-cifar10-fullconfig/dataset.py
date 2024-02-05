import torch.utils.data
import os
import cv2
import numpy as np
import torch.utils.data
from natsort import natsorted

class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, path, filenames, transform=None):
        self.path = path
        self.filenames = filenames
        self.transform = transform

        self.images = natsorted([os.path.join(self.path, 'images', filename) for filename in filenames])
        self.masks = natsorted([os.path.join(self.path, 'breast_masks', filename) for filename in filenames])
        self.contours = natsorted([os.path.join(self.path, 'dense_masks', filename) for filename in filenames])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Load image
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load breast mask and dense mask
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        contour = cv2.imread(self.contours[index], cv2.IMREAD_GRAYSCALE)

        # Combine masks if needed or process them to fit the AutoAlbument requirements
        # For example, stacking them along the depth if required.
        combined_mask = np.stack([mask, contour], axis=-1)

        if self.transform:
            transformed = self.transform(image=image, mask=combined_mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask
