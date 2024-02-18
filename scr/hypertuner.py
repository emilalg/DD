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
from utils import Config, load_config_from_args, load_config_from_env, get_augmentations
import optuna
import copy
import json

class hypertuner:

    config = Config()

    
    data = None
    DEVICE = None

    def __init__(self):   
        config = self.config
        augmentations = get_augmentations(config)

        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")

        torch.manual_seed(1990)

        train_set, val_set = get_dataset_splits(path=config.train_data_path, model_name=config.model_name)

        # create dataset and dataloader
        train_dataset = MammoDataset(
            path=config.train_data_path,
            filenames=train_set,
            augmentations=augmentations,
        )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=config.train_batch_size, num_workers=config.num_workers
        )
        # create validation dataset and dataloader
        valid_dataset = MammoDataset(
            path=config.train_data_path,
            filenames=val_set,
            augmentations=None,
        )
        valid_dataloader = DataLoader(
            valid_dataset, shuffle=True, batch_size=config.valid_batch_size, num_workers=config.num_workers
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
        config.optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        config.activation_function = trial.suggest_categorical("activation_function", ["sigmoid", "softmax"])
        config.learning_rate = trial.suggest_float('lr', 0.00001, 0.0001, log=True)

        if config.use_augmentation:
            # Only suggest these parameters if use_augmentation is True
            config.affine_translate_percent_x_limit = [
                trial.suggest_float('translate_percent_x_min', -0.20, 0.20),
                trial.suggest_float('translate_percent_x_max', -0.20, 0.20)
            ]
            config.affine_translate_percent_y_limit = [
                trial.suggest_float('translate_percent_y_min', -0.20, 0.20),
                trial.suggest_float('translate_percent_y_max', -0.20, 0.20)
            ]
            config.affine_shear_limit = [
                trial.suggest_float('shear_limit_min', -20, 20),
                trial.suggest_float('shear_limit_max', -20, 20)
            ]
            config.affine_rotate_limit = [
                trial.suggest_float('rotate_limit_min', -30, 30),
                trial.suggest_float('rotate_limit_max', -30, 30)
            ]
            # ALways apply augmentation. Should you even change probability other than 1?
            config.probability = 1
        # we set the loss function and parameters, and initialize the loss function in the config directly
        # ugly code but w.e
        # note the correct config name this time :)
        lossfn = trial.suggest_categorical("loss", ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss', 'FocalTverskyPlusPlusLoss', 'ComboLoss', 'FocalTverskyPlusPlusLoss'])
        if lossfn == 'DiceLoss':
            beta = trial.suggest_float('diceloss_beta', 0.1, 2, log=True)
            eps = trial.suggest_float('diceloss_eps', 0.1, 2, log=True)
            config.loss_function = smp.utils.losses.DiceLoss(beta=beta, eps=eps)
        elif lossfn == 'TverskyLoss':
            alpha = trial.suggest_float('tverskyloss_alpha', 0.1, 1, log=True)
            beta = trial.suggest_float('tverskyloss_beta', 0.1, 1, log=True)
            eps = trial.suggest_float('tverskyloss_eps', 0.1, 2, log=True)
            config.loss_function = smp.utils.losses.TverskyLoss(alpha=alpha, beta=beta, eps=eps)
        elif lossfn == 'FocalTverskyLoss':
            alpha = trial.suggest_float('focaltverskyloss_alpha', 0.1, 1, log=True)
            beta = trial.suggest_float('focaltverskyloss_beta', 0.1, 1, log=True)
            eps = trial.suggest_float('focaltverskyloss_eps', 0.1, 2, log=True)
            gamma = trial.suggest_float('focaltverskyloss_gamma', 0.1, 1, log=True)
            config.loss_function = smp.utils.losses.FocalTverskyLoss(alpha=alpha, beta=beta, eps=eps, gamma=gamma)
        elif lossfn == 'FocalTverskyPlusPlusLoss':
            alpha = trial.suggest_float('focaltverskyloss_alpha', 0.1, 1, log=True)
            beta = trial.suggest_float('focaltverskyloss_beta', 0.1, 1, log=True)
            eps = trial.suggest_float('focaltverskyloss_eps', 0.1, 2, log=True)
            gamma = trial.suggest_float('focaltverskyloss_gamma', 0.1, 1, log=True)
            config.loss_function = smp.utils.losses.FocalTverskyLoss(alpha=alpha, beta=beta, eps=eps, gamma=gamma)
        elif lossfn == 'ComboLoss':
            config.loss_function = smp.utils.losses.ComboLoss()
        elif lossfn == 'FocalTverskyPlusPlusLoss':
            # we leave eps as constant ?
            alpha = trial.suggest_float('focaltverskyplusplusloss_alpha', 0.1, 1, log=True)
            beta = trial.suggest_float('focaltverskyplusplusloss_beta', 0.1, 1, log=True)
            gamma = trial.suggest_float('focaltverskyplusplusloss_gamma', 0.1, 5, log=True)
            config.loss_function = smp.utils.losses.FocalTverskyPlusPlusLoss(alpha=alpha, beta=beta, gamma=gamma)

        # run trial
        out = self.__run(config)

        # save modified config, so we can convert it to a model later ( if its any good :) )
        trial.set_user_attr("config", config)

        # return float to optuna optimizer call
        return out["valid_logs"].pop()["l1_loss"]


    def __run(self, config: Config):

        DEVICE = self.DEVICE
        dl_train = self.data["train"]["dataloader"]
        dl_valid = self.data["valid"]["dataloader"]

        # create segmentation model with pretrained encoder
        model = getattr(smp, config.segmentation_model)(
            encoder_name=config.encoder,
            encoder_weights=config.pretrained_weights,
            classes=1,
            activation=config.activation_function,
        )

        model = model.to(DEVICE)
        model = nn.DataParallel(model)

        # define loss function (if loss_function not properly initialized use old code)
        # old: loss = getattr(smp.utils.losses, config["loss_fucntion"])()
        loss = config.loss_function
        if config.loss_function == None:
            Exception("Loss function not initialized")
        

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
                dict(params=model.parameters(), lr=config.learning_rate),
            ]
        )

        # LR_SCHEDULAR = 'steplr', 'reducelr', 'cosineannealinglr'
        # steplr: Decay the learning rate by gamma every step_size epochs.
        # reducelr: Reduce learning rate when a metric has stopped improving.
        # cosineannealinglr: Cosine annealing scheduler. if T_max (max_iter) is reached, the learning rate is annealed linearly to zero.
        # not in argparser, todo?
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

        print(f'Executing with config {config}')

        train_logs = []
        valid_logs = []

        for i in range(0, config.num_epochs):
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
        temp = self.__run(config) # ¯\_(ツ)_/¯
        torch.save(temp["model"], os.path.join(config.output_path, f"models/{config.model_name}.pth"))
        
        

def main():
    config = Config()
    
    # initalize hypertuner
    ht = hypertuner()
    num_trials = ht.config.num_trials
    print(f'\n\n****** Running {num_trials} trials ******\n\n')
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
    # because of the queued trial, n_trials should be the number you want to run +1
    study.optimize(ht.run_trial, n_trials=num_trials+1)
    print(f'Best parameters : {study.best_params}')
    
    # output all trials
    trials = study.get_trials()

    # write to log file
    outfile = open(os.path.join(config.output_path, f"logs/hypertuner.txt"), "w+") 
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