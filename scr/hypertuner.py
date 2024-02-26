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
from dataset import MammoDataset, MammoEvaluation, construct_val_set, get_dataset_splits
from predictions import load_ground_truths, process_testsubmission_mode  # for loading dataset
import segmentation_models_multi_tasking as smp  # for segmentation model
import matplotlib.pyplot as plt  # for plotting graphs
import os
from utils import Config, load_config_from_args, load_config_from_env
import optuna
import copy
import json
from trial_parameters import TrialParameters

class hypertuner:

    config = Config()
    trial_params = None 
    
    data = None
    DEVICE = None

    def __init__(self):   
        config = self.config
        self.trial_params = TrialParameters(loss=self.config.loss_function)

        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")
        print(f"Device: {self.DEVICE}")

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
        
        # predictions data initialization
        self.ground_truths = load_ground_truths(os.path.join(config.train_data_path, "../../train.csv"))
        ground_truths_path = os.path.join(config.train_data_path, "../../train.csv")
        test_dataset = MammoEvaluation(
            path=os.path.join(config.PROJECT_ROOT, config.train_data_path), mode=config.prediction_mode, ground_truths_path=ground_truths_path, model_name=config.model_name
        )
        self.predictions_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=config.num_workers)

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
        
    def suggest_loss_params(self, trial, lossfn):

        loss_name = lossfn

        alpha_param_name = f'{loss_name.lower()}_alpha'
        beta_param_name = f'{loss_name.lower()}_beta'

        alpha_value = trial.suggest_float(alpha_param_name, 0.1, 1, log=True)
       
        beta_min = 1 - alpha_value  
        beta_max = 1 - alpha_value  

        beta_value = trial.suggest_float(beta_param_name, beta_min, beta_max, log=True)
        
        loss_functions = {
            'Diceloss' : (smp.utils.losses.DiceLoss, lambda trial: {'beta': trial.suggest_float('diceloss_beta', 0.1, 1, log=True)}),
            'TverskyLoss' : (smp.utils.losses.TverskyLoss, lambda trial:
                            {'alpha': alpha_value, 
                            'beta': beta_value}),
            'FocalTverskyLoss': (smp.utils.losses.FocalTverskyLoss, lambda trial: 
                            {'alpha': alpha_value, 
                            'beta': beta_value, 
                            'gamma': trial.suggest_float('focaltverskyloss_gamma', 0.1, 3, log=True)}),
            'FocalTverskyPlusPlusLoss' : (smp.utils.losses.FocalTverskyPlusPlusLoss, lambda trial: 
                            {'alpha': alpha_value,
                            'beta':  beta_value,
                            'gamma': trial.suggest_float('focaltverskyloss_gamma', 0.1, 1, log=True)}),
            'ComboLoss' : (smp.utils.losses.ComboLoss, lambda trial: {}),
            'DSCPlusPlusLoss' : (smp.utils.losses.DSCPlusPlusLoss, lambda trial: 
                            {'beta': trial.suggest_float('dscplusplusloss_beta', 0.3, 1, log=True), 
                             'gamma': trial.suggest_float('dscplusplusloss_gamma', 2, 3, log=True)})
        }

        # Suggest parameters and instantiate the loss function
        if lossfn in loss_functions:
            loss_class, params_func = loss_functions[lossfn]
            params = params_func(trial)
            return loss_class(**params)

        raise ValueError(f"Unknown loss function: {lossfn}")

    def run_trial(self, trial):
        # deepcopy so that we do not accidentally modify base config
        config = copy.deepcopy(self.config)
        
        # Now we modify the hyper params in the config with Optuna
        # These are passed through the trial parameter
        config.optimizer = trial.suggest_categorical("optimizer", ["Adam"])
        config.activation_function = trial.suggest_categorical("activation_function", ["sigmoid"])

        lr_min, lr_max = self.trial_params.lr_min_max
        config.learning_rate = trial.suggest_float('lr', lr_min, lr_max, log=True)

        # we set the loss function and parameters, and initialize the loss function in the config directly
        # ugly code but w.e
        # note the correct config name this time :)
        lossfn = config.loss_function
        
        config.loss_function = self.suggest_loss_params(trial, lossfn)
        
        # run trial
        out = self.__run(config)
        # save modified config, so we can convert it to a model later ( if its any good :) )
        trial.set_user_attr("config", config)
        # print(f'\n\n out: {out} \n\n')
        # print(f'\n\n MAE: {out["mae"]} \n\n')

        # return float to optuna optimizer call
        return out["mae"]


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
            
        # run predictions
        mae = process_testsubmission_mode(self.predictions_dataloader, model, self.ground_truths)
        print(f'\n Mean Absolute Error: {mae} \n')

        out = {
            "train_logs": train_logs,
            "valid_logs": valid_logs,
            "model" : model,
            "mae": mae
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

    trial_params = ht.trial_params.get_trial_parameters()
 
    print(f'Adding enqueue trial with parameters: {trial_params}')
    study.enqueue_trial(trial_params)

    # some params to improve the search efficiency perhaps ? :)
    # defaults look ok
    # because of the queued trial, n_trials should be the number you want to run +1
    study.optimize(ht.run_trial, n_trials=num_trials)
    print('Optimization Complete. Saving and Exporting Best Model...')

    best_params = study.best_params
    best_trial_config = study.best_trial.user_attrs["config"]
    ht.export_model(best_trial_config)
    print(f'Best parameters : {best_params}')
    print('Best model saved successfully.')
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

    

    
if __name__ == "__main__":
    main()