# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:34:32 2020

train.py is used to train the model. It takes command line arguments and train the model.
It saves the logs and model in the specified path.

@author: rajgudhe
"""

# importing libraries
import argparse  # for command line arguments
import re
import torch  # for deep learning
import torch.optim as optim  # for optimization
import torch.nn as nn  # for neural network
from torch.utils.data import DataLoader  # for loading data
from dataset import MammoDataset, get_dataset_splits  # for loading dataset
import segmentation_models_multi_tasking as smp  # for segmentation model
import matplotlib.pyplot as plt  # for plotting graphs
import os
from utils import load_env


"""These are the default parameters. Do not modify them here. instead create .env file in project root or use command line arguments!"""
load_env()
## hyperparmetres
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
# LOGS_FILE_PATH = os.getenv("LOGS_FILE_PATH", "test_output/logs/unet.txt")
# MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "test_output/models/unet.pth")
MODEL_NAME = os.getenv("MODEL_NAME", "double_d")


print("Hyperparameters:")
print(f"  OPTIMIZER: {OPTIMIZER}")
print(f"  LOSS: {LOSS}")
print(f"  LR: {LR}")
print(f"  LR_SCHEDULE: {LR_SCHEDULE}")
print(f"  MODEL: {MODEL}")
print(f"  ENCODER: {ENCODER}")
print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
print(f"  NUM_WORKERS: {NUM_WORKERS}")
print(f"  TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}")
print(f"  VALID_BATCH_SIZE: {VALID_BATCH_SIZE}")
print(f"  ACTIVATION_FUNCTION: {ACTIVATION_FUNCTION}")
print(f"  PRETRAINED_WEIGHTS: {PRETRAINED_WEIGHTS}")
print(f"  DATA_PATH: {DATA_PATH}")
# print(f"  LOGS_FILE_PATH: {LOGS_FILE_PATH}")
# print(f"  MODEL_SAVE_PATH: {MODEL_SAVE_PATH}")

"""
Function to parse command line arguments

@returns: config object with all the command line arguments
"""


def main():
    def parse_args():
        parser = argparse.ArgumentParser()  # create an argumentparser object
        # add arguments to the parser object
        parser.add_argument("--data_path", default=DATA_PATH, type=str, help="dataset root path")
        parser.add_argument(
            "-tb",
            "--train_batch_size",
            default=TRAIN_BATCH_SIZE,
            type=int,
            metavar="N",
            help="mini-batch size (default: 4)",
        )
        parser.add_argument(
            "-vb",
            "--valid_batch_size",
            default=VALID_BATCH_SIZE,
            type=int,
            metavar="N",
            help="mini-batch size (default: 4)",
        )
        parser.add_argument(
            "--num_workers", default=NUM_WORKERS, type=int, help="Number of workers (default: 0)"
        )

        parser.add_argument(
            "--segmentation_model", default=MODEL, type=str, help="Segmentation model Unet/FPN"
        )
        parser.add_argument(
            "--encoder", default=ENCODER, type=str, help="encoder name resnet18, vgg16......."
        )  # change here
        parser.add_argument(
            "--pretrained_weights", default=PRETRAINED_WEIGHTS, type=str, help="imagenet weights"
        )
        parser.add_argument(
            "--activation_function",
            default=ACTIVATION_FUNCTION,
            type=str,
            help="activation of the final segmentation layer",
        )

        parser.add_argument("--loss_fucntion", default=LOSS, type=str, help="loss fucntion")
        parser.add_argument("--optimizer", default=OPTIMIZER, type=str, help="optimization")
        parser.add_argument("--lr", default=LR, type=float, help="initial learning rate")
        parser.add_argument("--num_epochs", default=NUM_EPOCHS, type=int, help="Number of epochs")

        parser.add_argument(
            "--model_name", default=MODEL_NAME, type=str, help="name of the model (used for saving all related data)"
        )  

        # parser.add_argument(
        #     "--logs_file_path", default=LOGS_FILE_PATH, type=str, help="path to save logs"
        # )  # change here
        # parser.add_argument(
        #     "--model_save_path", default=MODEL_SAVE_PATH, type=str, help="path to save the model"
        # )  # change her

        config = parser.parse_args()  # parse the arguments and store it in config object

        return config  # return the config object with all the command line arguments

    # Convert the config object to dictionary and print it
    config = vars(parse_args())
    print(config)

    # set random seeds for reproducibility
    torch.manual_seed(1990)

    train_set, val_set, test_set = get_dataset_splits(path=os.path.join(os.path.dirname(__file__), DATA_PATH), model_name=MODEL_NAME)

    # create dataset and dataloader
    train_dataset = MammoDataset(
        path=os.path.join(os.path.dirname(__file__), config['data_path']),
        filenames=train_set,
        augmentations=None,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config["train_batch_size"], num_workers=config["num_workers"]
    )

    # create validation dataset and dataloader
    valid_dataset = MammoDataset(
        path=os.path.join(os.path.dirname(__file__), config['data_path']),
        filenames=val_set,
        augmentations=None,
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=True, batch_size=config["valid_batch_size"], num_workers=config["num_workers"]
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
    model = getattr(smp, config["segmentation_model"])(
        encoder_name=config["encoder"],
        encoder_weights=config["pretrained_weights"],
        classes=1,
        activation=config["activation_function"],
    )

    model = model.to(DEVICE)
    model = nn.DataParallel(model)
    # print(model, (3, 256, 256))

    # define loss function
    loss = getattr(smp.utils.losses, config["loss_fucntion"])()

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
    optimizer = getattr(torch.optim, config["optimizer"])(
        [
            dict(params=model.parameters(), lr=config["lr"]),
        ]
    )

    # LR_SCHEDULAR = 'steplr', 'reducelr', 'cosineannealinglr'
    # steplr: Decay the learning rate by gamma every step_size epochs.
    # reducelr: Reduce learning rate when a metric has stopped improving.
    # cosineannealinglr: Cosine annealing scheduler. if T_max (max_iter) is reached, the learning rate is annealed linearly to zero.
    if LR_SCHEDULE == "steplr":
        lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    elif LR_SCHEDULE == "reducelr":
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
    with open(f"test_output/logs/{MODEL_NAME}.txt", "a+") as logs_file:
        print('Epoch \t Loss Function \t Train Logs \t Valid Logs', file=logs_file)
        max_score = 0
        for i in range(0, config["num_epochs"]):
            print("\nEpoch: {}".format(i))
            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

             # Generate the dynamic loss key
            loss_key = get_loss_key(config["loss_fucntion"])


            train_accuracy.append(train_logs["accuracy"])
            valid_accuracy.append(valid_logs["accuracy"])

            # Append loss values dynamically
            train_loss.append(train_logs[loss_key])
            valid_loss.append(valid_logs[loss_key])

            # Print train and validation logs
            log_line = f"{config['loss_fucntion']}:"
            for key, value in train_logs.items():
                log_line += f"\t {key} - {value}"
            for key, value in valid_logs.items():
                log_line += f"\t {key} - {value}"
            log_line += f"\t Epoch: {i}"
            print(log_line, file=logs_file)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs["iou_score"]:
                max_score = valid_logs["iou_score"]
                torch.save(model, f"test_output/models/{MODEL_NAME}.pth")
                print("Model saved!")

    # Plot accuracy and loss curves
    epochs = range(1, config["num_epochs"] + 1)

    print(len(epochs), len(train_loss), len(valid_loss))
    # Plot accuracy
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, valid_accuracy, label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    #plt.show() # why are these blocking?
    plt.savefig(f"test_output/logs/{MODEL_NAME}_accuracy.jpg")

    # Plot loss
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    #plt.show()
    plt.savefig(f"test_output/logs/{MODEL_NAME}_loss.jpg")
    
def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def get_loss_key(loss_name):
    key_base = camel_to_snake(loss_name[:-4])
    return key_base + "_loss_weighted"

if __name__ == "__main__":
    main()
