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
DEVICE = 'cuda'
# Create evaluation object
test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

# 5. Run the evaluation on the test set and save the results in a text file
with open(f"test_output/evaluation/{MODEL_NAME}.txt", 'a+') as logs_file:
    for i in range(0, config['num_epochs']):
        print('\nEpoch: {}'.format(i))

        test_logs = test_epoch.run(test_dataloader)

        print('{} \t {} \t {} \t {} \t {} \t {}'.format(i,
            test_logs['precision'],
            test_logs['recall'],
            test_logs['fscore'],
            test_logs['accuracy'],
            test_logs['iou_score']), file=logs_file)