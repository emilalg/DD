import torch.utils.data
import os
import cv2
import numpy as np
import torch.utils.data
from natsort import natsorted

MODEL_NAME = os.getenv("MODEL_NAME", "double_d")

class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, path=0, filenames=0, transform=None):
        self.path = path
        self.path = os.path.join(os.path.dirname(__file__), f"../../test_output/logs/{MODEL_NAME}_splits.txt")
        self.filenames = filenames
        self.filenames= ["00a6b0d56eb5136c1be2c3d624b04dad.jpg"]
        self.transform = transform
        print(f"Path: {path}")
        

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
