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
import optuna
import copy
import json

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

"""
Comment out print stuff :D
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

"""
Function to parse command line arguments

@returns: config object with all the command line arguments
"""

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


class hypertuner:

    config = vars(parse_args())
    data = None
    DEVICE = None

    def __init__(self):   
        config = self.config
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")

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

        self.data = {
            "train": {
                "dataset": train_dataset,
                "dataloader": train_dataloader
            },
            "valid": {
                "dataset": valid_dataset,
                "dataloader": valid_dataloader
            }
        }


    def run_trial(self, trial):
        # deepcopy so that we do not accidentally modify base config
        config = copy.deepcopy(self.config)
        
        # Now we modify the hyper params in the config with Optuna
        # These are passed through the trial parameter
        config["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        config["activation_function"] = trial.suggest_categorical("activation_function", ["sigmoid", "softmax"])
        config["lr"] = trial.suggest_float('lr', 0.00001, 0.0001, log=True)

        # we set the loss function and parameters, and initialize the loss function in the config directly
        # ugly code but w.e
        # note the correct config name this time :)
        lossfn = trial.suggest_categorical("loss", ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss'])
        if lossfn == 'DiceLoss':
            beta = trial.suggest_float('diceloss_beta', 0.1, 2, log=True)
            eps = trial.suggest_float('diceloss_eps', 0.1, 2, log=True)
            config["loss_function"] = smp.utils.losses.DiceLoss(beta=beta, eps=eps)
        elif lossfn == 'TverskyLoss':
            alpha = trial.suggest_float('tverskyloss_alpha', 0.1, 1, log=True)
            beta = trial.suggest_float('tverskyloss_beta', 0.1, 1, log=True)
            eps = trial.suggest_float('tverskyloss_eps', 0.1, 2, log=True)
            config["loss_function"] = smp.utils.losses.TverskyLoss(alpha=alpha, beta=beta, eps=eps)
        elif lossfn == 'FocalTverskyLoss':
            alpha = trial.suggest_float('focaltverskyloss_alpha', 0.1, 1, log=True)
            beta = trial.suggest_float('focaltverskyloss_beta', 0.1, 1, log=True)
            eps = trial.suggest_float('focaltverskyloss_eps', 0.1, 2, log=True)
            gamma = trial.suggest_float('focaltverskyloss_gamma', 0.1, 1, log=True)
            config["loss_function"] = smp.utils.losses.FocalTverskyLoss(alpha=alpha, beta=beta, eps=eps, gamma=gamma)

        # run trial
        out = self.__run(config)

        # save config, so we can convert it to a model later ( if its any good :) )
        trial.set_user_attr("config", out)

        # return float to optuna optimizer call
        return out["valid_logs"].pop()["l1_loss"]


    def __run(self, config):

        DEVICE = self.DEVICE
        dl_train = self.data["train"]["dataloader"]
        dl_valid = self.data["valid"]["dataloader"]

        # create segmentation model with pretrained encoder
        model = getattr(smp, config["segmentation_model"])(
            encoder_name=config["encoder"],
            encoder_weights=config["pretrained_weights"],
            classes=1,
            activation=config["activation_function"],
        )

        model = model.to(DEVICE)
        model = nn.DataParallel(model)

        # define loss function (if loss_function not properly initialized use old code)
        # old: loss = getattr(smp.utils.losses, config["loss_fucntion"])()
        loss = config["loss_function"]
        if config["loss_function"] == None:
            print('config["loss_function"] not set :(')
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
        # not in argparser, todo?
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

        print(f'Executing with config {config}')

        train_logs = []
        valid_logs = []

        for i in range(0, config["num_epochs"]):
            print("\nEpoch: {}".format(i))
            train_logs.append(train_epoch.run(dl_train))
            valid_logs.append(valid_epoch.run(dl_valid))

        out = {
            "train_logs": train_logs,
            "valid_logs": valid_logs,
            "model" : model
        }
        return out
    
    def export_model(self, config):
        # train and save model based on a config
        print(f'Exporting model with config {config}')
        temp = self.__run(config)
        torch.save(temp["model"], f"test_output/models/{MODEL_NAME}.pth")
        
        

def main():
    # initalize hypertuner
    ht = hypertuner()

    # we can make a study deterministic by assigning a custom sampler with a set seed
    # does not feel necessary atm
    study = optuna.create_study(direction='minimize')

    # queue run wtih default parameters
    study.enqueue_trial({
        "optimizer": "Adam",
        "loss": "FocalTverskyLoss",
        "activation_function": "sigmoid",
        "focaltverskyloss_alpha": 0.3,
        "focaltverskyloss_beta": 0.7,
        "focaltverskyloss_eps": 1.0,
        "focaltverskyloss_gamma": 0.75,
        "lr": 0.0001
    })

    # some params to improve the search efficiency perhaps ? :)
    # defaults look ok
    # because queued trial n_trials should be the number you want to run +1
    study.optimize(ht.run_trial, n_trials=1)
    print(f'Best parameters : {study.best_params}')
    
    # output all trials
    trials = study.get_trials()

    # write to log file
    outfile = open(f"test_output/logs/hypertuner.txt", "w+") 
    outfile.write(f'Best parameters: {json.dumps(study.best_params, default=str, indent=4, sort_keys=True)} \n\n')
    for trial in trials:
        out = {
            "trial_nro": trial.number,
            "L1Loss": trial.value,
            "parameters" : trial.params
        }
        outfile.write(json.dumps(out, default=str, indent=4, sort_keys=True))
        outfile.write('\n')
    outfile.close()

    # write best model to file
    print('Saving best model')
    ht.export_model(study.best_trial.user_attrs["config"])

    

    
if __name__ == "__main__":
    main()
