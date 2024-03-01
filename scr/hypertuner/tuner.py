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
from .runner import Runner
from utils import makedirs
import pickle

class Tuner:

    config = Config()
    runner = Runner(config)
    study = None
    trial_params = None 
    output_path = None
    direction = 'minimize'
    load_type = 'file'

    temp_model = None

    def __init__(self, load_type='file', direction='minimize'):   
        
        self.direction = direction
        self.load_type = load_type
        self.output_path = f'{self.config.output_path}/hypertuner/{self.config.study_name}'
        makedirs(f'{self.output_path}/')
        self.trial_params = TrialParameters(loss=self.config.loss_function)
        torch.manual_seed(1990)
        test = os.path.join(self.output_path, "/hypertuner.txt")
        print(f'agony {self.output_path + "/hypertuner.txt"}')

    def create_study(self) -> optuna.Study:
        self.study = optuna.create_study(direction=self.direction, pruner=optuna.pruners.MedianPruner(n_startup_trials=1,n_warmup_steps=1, interval_steps=2))
        trial_params = self.trial_params.get_trial_parameters()
        print(f'Adding enqueue trial with parameters: {trial_params}')
        self.study.enqueue_trial(trial_params)
        return self.study
    
    def load_study(self) -> bool:
        if self.load_type == 'file':
            try:
                with open(f'{self.output_path}/study.pkl', 'rb') as f:
                    self.study = pickle.load(f)
                    print('Study loaded.')
                    return True
            except:
                print('Study not found.')
                return False
        else:
            print('Sql loading not implemented yet')
            return False

    """
    Use this.
    """
    def get_study(self) -> optuna.Study:
        if self.study != None:
            return self.study
        if self.load_study():
            return self.study
        return self.create_study()


    def callback(self, study, trial):
        print('Callback.')
        with open(f'{self.output_path}/study.pkl', 'wb') as f:
            pickle.dump(study, f)
        
        self.export_logs()

        if study.best_trial.number == trial.number:
            # saving best model
            self.export_model()


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
        out = self.runner.run(config,trial)
        # save modified config, so we can convert it to a model later ( if its any good :) )
        trial.set_user_attr("config", config)
        # print(f'\n\n out: {out} \n\n')
        # print(f'\n\n MAE: {out["mae"]} \n\n')

        # save temp model
        self.temp_model = out["model"]

        # return float to optuna optimizer call
        return out["mae"]

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
    
    def export_model(self):
        # train and save model based on a config
        print(f'Saving model')
        torch.save(self.temp_model, f"{self.output_path}/{self.config.model_name}.pth")

    def export_logs(self):
        study = self.study
        trials = study.get_trials()

        # write to log file
        outfile = open(f"{self.output_path}/hypertuner.txt", "w+") 
        outfile.write(f'Best parameters: {json.dumps(study.best_params, default=str, indent=4, sort_keys=True)} \n\n')
        for trial in trials:
            out = {
                "trial_nro": trial.number,
                "Mae": trial.value,
                "parameters" : trial.params
            }
            outfile.write(json.dumps(out, default=str, indent=4, sort_keys=True))
            outfile.write('\n')
        outfile.close()