# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:34:32 2020

train.py is used to train the model. It takes command line arguments and train the model.
It saves the logs and model in the specified path.

@author: rajgudhe
"""

# importing libraries
import argparse  # for command line arguments
import torch  # for deep learning
import torch.optim as optim  # for optimization
import torch.nn as nn  # for neural network
from torch.utils.data import DataLoader  # for loading data
from dataset import MammoDataset, get_dataset_splits  # for loading dataset
import segmentation_models_multi_tasking as smp  # for segmentation model
import matplotlib.pyplot as plt  # for plotting graphs
import os
import datetime
from utils import Config, get_loss_key, load_config_from_args, load_config_from_env



"""
Function to parse command line arguments

@returns: config object with all the command line arguments
"""


def main():
    config = Config()
    print(f"train.py config:\n ", config)
    # set random seeds for reproducibility
    torch.manual_seed(1990)

    train_set, val_set = get_dataset_splits(path=config.train_data_path, model_name=config.model_name)

    # create dataset and dataloader
    train_dataset = MammoDataset(
        path=config.train_data_path,
        filenames=train_set,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.train_batch_size, num_workers=config.num_workers
    )

    # create validation dataset and dataloader
    valid_dataset = MammoDataset(
        path=config.train_data_path,
        filenames=val_set,
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=True, batch_size=config.valid_batch_size, num_workers=config.num_workers
    )

    # define device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #    DEVICE = torch.device("mps")
    #    PYTORCH_ENABLE_MPS_FALLBACK=1
    else:
        DEVICE = torch.device("cpu")
    print(DEVICE)

    # create segmentation model with pretrained encoder
    model = getattr(smp, config.segmentation_model)(
        encoder_name=config.encoder,
        encoder_weights=config.pretrained_weights,
        classes=1,
        activation=config.activation_function,
    )

    model = model.to(DEVICE)
    model = nn.DataParallel(model)
    # print(model, (3, 256, 256))

    # define loss function
    loss = getattr(smp.utils.losses, config.loss_function)()

    # define metrics which will be monitored during training
    metrics = [
        smp.utils.metrics.L1Loss(),
        smp.utils.metrics.Precision(),
        smp.utils.metrics.Recall(),
        smp.utils.metrics.Accuracy(),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.IoU(threshold=0.5),
        # calculate mean absolute error (important)
    ]

    # define optimizer and lr scheduler which will be used during training
    optimizer = getattr(torch.optim, config.optimizer)(
        [
            #AdamW parameters is {betas: tuple = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0001}
            dict(params=model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay),
        ]
    )

    # LR_SCHEDULAR = 'steplr', 'reducelr', 'cosineannealinglr'
    # steplr: Decay the learning rate by gamma every step_size epochs.
    # reducelr: Reduce learning rate when a metric has stopped improving.
    # cosineannealinglr: Cosine annealing scheduler. if T_max (max_iter) is reached, the learning rate is annealed linearly to zero.
    if config.learning_scheduler == "steplr":
        lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    elif config.learning_scheduler == "reducelr":
        lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
    else:
        lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)



    # create training epoch runners
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

    # create validation epoch runner
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # creating lists to store accuracy and loss values
    train_accuracy = []
    valid_accuracy = []
    train_loss = []
    valid_loss = []


    # open the logs file
    with open(os.path.join(config.output_path, f"logs/{config.model_name}.txt"), "a+") as logs_file:
        print('Epoch \t Loss Function \t Train Logs \t Valid Logs', file=logs_file)
        max_score = 0
        print(config.num_epochs)
        for i in range(0, config.num_epochs):
            print("\nEpoch: {}".format(i))
            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

             # Generate the dynamic loss key
            loss_key = get_loss_key(config.loss_function)


            train_accuracy.append(train_logs["accuracy"])
            valid_accuracy.append(valid_logs["accuracy"])

            # Append loss values dynamically
            train_loss.append(train_logs[loss_key])
            valid_loss.append(valid_logs[loss_key])

            # Print train and validation logs
            log_line = f"{config.loss_function}:"
            for key, value in train_logs.items():
                log_line += f"\t {key} - {value}"
            for key, value in valid_logs.items():
                log_line += f"\t {key} - {value}"
            log_line += f"\t Epoch: {i}"
            print(log_line, file=logs_file)

            # Save best model
            if max_score < valid_logs["fscore"]:
                max_score = valid_logs["fscore"]
                torch.save(model, f"test_output/models/{config.model_name}.pth")  # Save model state dict for best performance
                print(f"Best model saved at epoch {i}!")

    # Plot accuracy and loss curves
    epochs = range(1, config.num_epochs + 1)

    timestamp = datetime.datetime.now().strftime("%d-%m_%H-%M")

    accuracyDir = "test_output/logs/accuracy/"
    lossDir = "test_output/logs/loss"
    if not os.path.exists(accuracyDir):
        os.makedirs(accuracyDir)
    if not os.path.exists(lossDir):
        os.makedirs(lossDir)

    print(len(epochs), len(train_loss), len(valid_loss))
    # Plot accuracy
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, valid_accuracy, label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    # Savefig needs to be called first before show since closing the plt.show() image causes it to be freed from memory
    plt.savefig(f"test_output/logs/accuracy/{config.model_name}_accuracy_{timestamp}.jpg")
    #plt.show()
    # Plot loss
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(f"test_output/logs/loss/{config.model_name}_loss_{timestamp}.jpg")
    #plt.show()
    

if __name__ == "__main__":
    main()