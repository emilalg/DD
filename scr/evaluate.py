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

LOSS = 'FocalTverskyLoss' # change the results log ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss', 'BCEWithLogitsLoss']

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
    parser.add_argument('--data_path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='dataset_name', type=str, help='Mammogram view')
    parser.add_argument('-tb', '--test_batch_size', default=1, type=int, metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--loss_fucntion',default=LOSS, type=str, help='loss fucntion')
    parser.add_argument('--model_save_path', default='test_output/models/unet.pth', type=str, help='path to save the model')  # change here

    parser.add_argument('--results_path', default='test_output/evaluation/dataset_name.txt', type=str,
                        help='path to save the model')  # change here

    config = parser.parse_args()

    return config

# 1. Parse the CLI argument using parse_args()
config = vars(parse_args())
print(config)

# 2. First get the path to the dataset, then load the dataset using Torch's DataLoader
test_dataset = MammoDataset(path=config['data_path'], dataset=config['dataset'], split='test',augmentations=None)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size =1, num_workers=config['num_workers'])

# 3. Load the trained model
# load best saved checkpoint
model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module) 

# 4. Define the loss function and the metrics.
loss = getattr(smp.utils.losses, config['loss_fucntion'])()

# metrics are used to evaluate the model performance
metrics = [
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

# 5. Run the evaluation on the test set and save the results in a text file (specified by results_path)
with open(config['results_path'], 'a+') as logs_file:
    for i in range(0, config['num_epochs']):
        print('\nEpoch: {}'.format(i))

        test_logs = test_epoch.run(test_dataloader)

        print('{} \t {} \t {} \t {} \t {} \t {}'.format(i,
            test_logs['precision'],
            test_logs['recall'],
            test_logs['fscore'],
            test_logs['accuracy'],
            test_logs['iou_score']), file=logs_file)