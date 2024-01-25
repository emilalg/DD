# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:34:32 2020

@author: rajgudhe
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MammoDataset
import segmentation_models_multi_tasking as smp
import matplotlib.pyplot as plt



## hyperparmetres 
OPTIMIZER = 'Adam'
LOSS = 'FocalTverskyLoss' # change the results log ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss', 'BCEWithLogitsLoss']
LR = 0.0001
LR_SCHEDULE = 'reducelr' # 'steplr', 'reducelr'
MODEL = 'Unet' #'Unet'
ENCODER = 'resnet101'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='dataset_name', type=str, help='Mammogram dataset names [CMMD, DDSM, VINDR]')
    parser.add_argument('-tb', '--train_batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-vb', '--valid_batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    
    parser.add_argument('--segmentation_model', default=MODEL, type=str, help='Segmentation model Unet/FPN')
    parser.add_argument('--encoder', default=ENCODER, type=str, help='encoder name resnet18, vgg16.......')   # change here
    parser.add_argument('--pretrained_weights', default=None, type=str, help='imagenet weights')
    parser.add_argument('--activation_function',default='sigmoid', type=str, help='activation of the final segmentation layer')
    
    parser.add_argument('--loss_fucntion',default=LOSS, type=str, help='loss fucntion')
    parser.add_argument('--optimizer', default=OPTIMIZER, type=str, help='optimization')
    parser.add_argument('--lr', default=LR, type=float, help='initial learning rate')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs')
    
    parser.add_argument('--logs_file_path', default='test_output/logs/unet.txt', type=str, help='path to save logs') # change here
    parser.add_argument('--model_save_path', default='test_output/models/unet.pth', type=str, help='path to save the model') # change her
        
    config = parser.parse_args()

    return config


config = vars(parse_args())
print(config)   

torch.manual_seed(1990)

train_dataset = MammoDataset(path=config['data_path'], dataset=config['dataset'], split='train', augmentations=None)
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size =config['train_batch_size'], num_workers=config['num_workers'])

valid_dataset = MammoDataset(path=config['data_path'], dataset=config['dataset'], split='valid', augmentations=None)
print(len(valid_dataset))
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=config['valid_batch_size'], num_workers=config['num_workers'])


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create segmentation model with pretrained encoder

model = getattr(smp, config['segmentation_model'])(
    encoder_name= config['encoder'],
    encoder_weights = config['pretrained_weights'], 
    classes =1,
    activation = config['activation_function'])

model = model.to(DEVICE)
model = nn.DataParallel(model)
print(model, (3, 256, 256))


loss = getattr(smp.utils.losses, config['loss_fucntion'])()

metrics = [
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.IoU(threshold=0.5),
    
]


optimizer = getattr(torch.optim, config['optimizer'])([ 
    dict(params=model.parameters(), lr=config['lr']),
])


if LR_SCHEDULE == 'steplr':
    lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif LR_SCHEDULE == 'reducelr':
    lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
else:
    lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 2)

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    lr_schedular=lr_schedular,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []


with open(config['logs_file_path'], 'a+') as logs_file:
    # train model for 40 epochs
    
    max_score = 0
    for i in range(0, config['num_epochs']):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)


        train_accuracy.append(train_logs['accuracy'])
        valid_accuracy.append(valid_logs['accuracy'])
        train_loss.append(train_logs['focal_tversky_loss_weighted'])
        valid_loss.append(valid_logs['focal_tversky_loss_weighted'])

        print('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}'.format(i, train_logs['focal_tversky_loss_breast'],
                                                                                                                    train_logs['focal_tversky_loss_dense'],
                                                                                                                    train_logs['focal_tversky_loss_weighted'],
                                                                                                                    train_logs['precision'],
                                                                                                                    train_logs['recall'],
                                                                                                                    train_logs['accuracy'],
                                                                                                                    train_logs['fscore'],
                                                                                                                    train_logs['iou_score'],
                                                                                                                    valid_logs['focal_tversky_loss_breast'],
                                                                                                                    valid_logs['focal_tversky_loss_dense'],
                                                                                                                    valid_logs['focal_tversky_loss_weighted'],
                                                                                                                    valid_logs['precision'],
                                                                                                                    valid_logs['recall'],
                                                                                                                    valid_logs['accuracy'],
                                                                                                                    valid_logs['fscore'],
                                                                                                                    valid_logs['iou_score']), file=logs_file)

        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, config['model_save_path'])
            print('Model saved!')
            


# Plot accuracy and loss curves
epochs = range(1, config['num_epochs']+1)

# Plot accuracy
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, valid_accuracy, label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# Plot loss
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, valid_loss, label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()