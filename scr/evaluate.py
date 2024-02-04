# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:09:58 2020

evaluate.py is used to evaluate the results of the trained model on the test set.
Note that validation and training sets are not used here. evaluate.py requires a CUDA enabled GPU to run.

@author: rajgudhe
"""

import argparse
import torch
import torch.nn as nn
from dataset import MammoDataset
from torch.utils.data import DataLoader
import segmentation_models_multi_tasking as smp
from sklearn.metrics import mean_absolute_error
import os
from utils import load_env
from dataset import construct_test_set
import re
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()

LOSS = 'FocalTverskyLoss' # change the results log ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss', 'BCEWithLogitsLoss']


load_env()

OPTIMIZER = os.getenv("OPTIMIZER", "Adam")  # change the results log ['Adam', 'SGD']
LOSS = os.getenv("LOSS", "FocalTverskyLoss")  # change the results log ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss', 'BCEWithLogitsLoss, DSCPlusPlusLoss']
LR = float(os.getenv("LR", 0.0001))  # change the results log [0.0001, 0.00001]
LR_SCHEDULE = os.getenv("LR_SCHEDULE", "reducelr")  # 'steplr', 'reducelr'
MODEL = os.getenv("MODEL", "Unet")  # 'Unet'
ENCODER = os.getenv("ENCODER", "resnet101")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 4))
VALID_BATCH_SIZE = int(os.getenv("VALID_BATCH_SIZE", 4))
ACTIVATION_FUNCTION = os.getenv("ACTIVATION_FUNCTION", "sigmoid")  # 'sigmoid', 'softmax'
PRETRAINED_WEIGHTS = os.getenv("PRETRAINED_WEIGHTS", None)
DATA_PATH = os.getenv("DATA_PATH", "../breast-density-prediction/train/train")
LOGS_FILE_PATH = os.getenv("LOGS_FILE_PATH", "test_output/logs/unet.txt")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "test_output/models/unet.pth")
TEST_BATCH_SIZE = int(os.getenv("TEST_BATCH_SIZE", 4))
MODEL_NAME = os.getenv("MODEL_NAME", "double_d")

"""Parse the CLI arguments of the program. 
--data_path: root path to the directory containing the dataset
--dataset: name of the dataset directory
--tb, --test_batch_size: 
--num_epochs: number of epochs used for validation. Default is 10.
--num_workers: Number of worker processes for the dataloader. Default 0 is which means that the data will be loaded in the main process. 
--loss_fucntion: Loss function used for validation. Default is FocalTverskyLoss. Valid options are ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss', 'BCEWithLogitsLoss']
--model_save_path: path to the saved model used for evaluation
--results_path: path to the file where the results will be saved

@returns: The parsed arguments.
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=DATA_PATH, type=str, help='dataset root path')
    parser.add_argument('-tb', '--test_batch_size', default=TEST_BATCH_SIZE, type=int, metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--num_epochs', default=NUM_EPOCHS, type=int, help='Number of epochs')
    parser.add_argument('--num_workers', default=NUM_WORKERS, type=int, help='Number of workers')
    parser.add_argument('--loss_fucntion',default=LOSS, type=str, help='loss fucntion')
    #parser.add_argument('--model_save_path', default='test_output/models/unet.pth', type=str, help='path to save the model')  # change here

    parser.add_argument(
            "--model_name", default=MODEL_NAME, type=str, help="name of the model (used for saving all related data)"
        )  
    # parser.add_argument('--results_path', default='test_output/evaluation/dataset_name.txt', type=str,
    #                     help='path to save the model')  # change here

    config = parser.parse_args()

    return config

# 1. Parse the CLI argument using parse_args()
config = vars(parse_args())
print(config)

# 2. First get the path to the dataset, then load the dataset using Torch's DataLoader
# get the test set
test_set = construct_test_set(path=config['data_path'], model_name=config['model_name'])

test_dataset = MammoDataset(
        path=os.path.join(os.path.dirname(__file__), config['data_path']),
        filenames=test_set,
        augmentations=None,
    )
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size =1, num_workers=config['num_workers'])

# 3. Load the trained model
# load best saved checkpoint
model = torch.load(os.path.join(os.path.dirname(__file__), f'../test_output/models/{MODEL_NAME}.pth'))
model = nn.DataParallel(model.module)


# 4. Define the loss function and the metrics.
loss = getattr(smp.utils.losses, config['loss_fucntion'])()

# metrics are used to evaluate the model performance
metrics = [
    smp.utils.metrics.L1Loss(),
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.IoU(threshold=0.5),

]

# Device used for training. CUDA used an NVIDIA GPU and requires CUDA to be installed. CPU is not available here.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
#    DEVICE = torch.device("mps")
#    PYTORCH_ENABLE_MPS_FALLBACK=1
else:
    DEVICE = torch.device("cpu")
print(DEVICE)
    
    
# Create evaluation object
test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def get_loss_key(loss_name):
    key_base = camel_to_snake(loss_name[:-4])  # Remove 'Loss' and convert
    return key_base + "_loss_weighted"  # Append '_loss_weighted' as in train.py logs

# Convert the image to a pytorch tensor
def image_tensor(img):
    """
    Converts a PIL Image or NumPy array to a PyTorch tensor.
    Args:
    - img: PIL.Image or np.ndarray, the image to be converted to a PyTorch tensor.
    Returns:
    - image: torch.Tensor, the converted PyTorch tensor.
    Raises:
    - TypeError: if the input image is not a PIL.Image or np.ndarray.
    """
    
    if type(img) not in [np.ndarray, Image.Image]:
        raise TypeError("Input must be np.ndarray or PIL.Image")

    # Define a PyTorch tensor transformer pipeline
    torch_tensor = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    if type(img) == Image.Image:
        # Convert PIL image to PyTorch tensor
        image = torch_tensor(img)
        image = image.unsqueeze(0)
        print("tensor", type(image))
        return image
    elif type(img) == np.ndarray:
        # Convert NumPy array to RGB PIL image and then to PyTorch tensor
        pil_image = Image.fromarray(img).convert("RGB")
        image = torch_tensor(pil_image)
        image = image.unsqueeze(0)
        print("tensor", type(image))
        return image
    else:
        raise TypeError("Input must be np.ndarray or PIL.Image")


def mask_to_rgba(mask, color="red", opacity=0.5):
    MASK_COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    assert color in MASK_COLORS
    assert mask.ndim == 3 or mask.ndim == 2

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones * opacity), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones * opacity), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones * opacity), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones * opacity), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones * opacity), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones * opacity), axis=-1)
    
def visualize_segmentation_results(image_tensor, model, threshold=0.5):
    """
    Visualizes the segmentation results by the model with thresholding and color overlay.
    
    Parameters:
    - image_tensor: A PyTorch tensor of the image to be visualized. 
                    It should be in the shape of [C, H, W] and on the correct device.
    - model: The PyTorch model used for prediction.
    - threshold: The threshold for converting probability maps to binary masks.
    
    The function assumes that the model's prediction method returns two tensors: pred1 and pred2,
    which are the model's segmentation masks.
    """
    
    print(f"Visualizing. Image shape: {image_tensor.shape}, Model type: {type(model)}")

    # Set the model to evaluation mode
    model.eval()

    # Use 'with torch.no_grad()' to avoid tracking history in autograd
    with torch.no_grad():
        # Forward pass through the model
        predictions = model(image_tensor.unsqueeze(0))  # Add a batch dimension if needed
        
        pred1, pred2 = predictions

        # Apply threshold to convert predictions to binary masks
        pred1_binary = (pred1 > threshold).float()
        pred2_binary = (pred2 > threshold).float()

    # Convert the binary masks to RGBA where the mask is colored and the rest is transparent
    pred1_colored = mask_to_rgba(pred1_binary.squeeze().cpu().numpy(), color="green")
    pred2_colored = mask_to_rgba(pred2_binary.squeeze().cpu().numpy(), color="red")

    # Convert the original image tensor to PIL for consistent visualization
    input_pil = to_pil_image(image_tensor.cpu())

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(input_pil, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(input_pil, cmap='gray')
    axes[1].imshow(pred1_colored, interpolation='none')
    axes[1].set_title('Prediction 1 Overlay, Brest Tissue Mask')
    axes[1].axis('off')

    axes[2].imshow(input_pil, cmap='gray')
    axes[2].imshow(pred2_colored, interpolation='none')
    axes[2].set_title('Prediction 2 Overlay, Density Mask')
    axes[2].axis('off')

    plt.show()

    
# Define the maximum number of images to show
max_images_to_show = 5  # You can change this number as needed

with open(f"test_output/evaluation/{MODEL_NAME}.txt", 'a+') as logs_file:
    for i, batch_data in enumerate(test_dataloader):
        if i >= max_images_to_show:
            break  # Stop displaying images after reaching the maximum
        print(f"Processing batch {i+1}/{len(test_dataloader)}...")
        try:
            # Directly unpack the batch_data tuple into images, masks, and contours
            images, masks, contours = batch_data
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) # If needed
            contours = contours.to(DEVICE)

            # Make predictions
            with torch.no_grad():
                print("Making predictions...")

                pred1, pred2 = model(images)
                print("Predictions made.")

            # Visualize segmentation results for each batch
            print("Visualizing results...")
            for img in images:
                visualize_segmentation_results(img.cpu(), model)  # Update this function as needed
            print("Visualizations completed.")

        except Exception as e:
            print(f"Error during evaluation: {e}")

        # After all batches are processed, evaluate the performance on the entire test set
        if i == len(test_dataloader) - 1:
            print("Running final evaluation...")
            test_logs = test_epoch.run(test_dataloader)
            print("Final evaluation completed.")

            loss_key = get_loss_key(config['loss_fucntion'])
            log_line = f"{loss_key}:"
            for key, value in test_logs.items():
                log_line += f"\t {key} - {value}"
            log_line += f"\t Epoch: {config['num_epochs']}"

            print(log_line, file=logs_file)

print("Evaluation complete.")