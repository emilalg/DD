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
from tunerlogging import export_logs
from trial_O_builder_O_ import TrialBuilder

class Tuner:

    config = Config()
    runner = Runner(config)
    study = None
    trial_params = None 
    output_path = None
    direction = 'minimize'
    load_type = 'file'
    db_url = None

    temp_model = None

    def __init__(self, direction='minimize'):   
        
        self.direction = direction

        self.load_type = self.config.load_type
        self.db_url = self.config.db_url  

        self.output_path = f'{self.config.output_path}/hypertuner/{self.config.study_name}'
        makedirs(f'{self.output_path}/')
        self.trial_params = TrialParameters(loss=self.config.loss_function)
        self.trial_builder = TrialBuilder(self.config)
        torch.manual_seed(1990)
        test = os.path.join(self.output_path, "/hypertuner.txt")

    def create_study(self) -> optuna.Study:
        if self.load_type == 'sql' and self.db_url:
            self.study = optuna.create_study(direction=self.direction, 
                                             pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=self.config.pruning_interval), 
                                             study_name=self.config.study_name, 
                                             storage=self.db_url,
                                             load_if_exists=True)
            print('Created study with sql')
        else:
            self.study = optuna.create_study(direction=self.direction, 
                                             pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=self.config.pruning_interval), 
                                             study_name=self.config.study_name)
            print('Created study with file')
        trial_params = self.trial_params.get_trial_parameters()
        print(f'Adding enqueue trial with parameters: {trial_params}')
        self.study.enqueue_trial(trial_params)
        return self.study
    
    def load_study(self) -> bool:
        if self.load_type == 'file':
            try:
                with open(f'{self.output_path}/{self.config.model_name}.pkl', 'rb') as f:
                    self.study = pickle.load(f)
                    print('Study loaded.')
                    return True
            except:
                print('Study not found.')
                return False
        elif self.load_type == 'sql' and self.db_url:
            try:
                self.study = optuna.load_study(study_name=self.config.study_name, storage=self.db_url)
                print('Study loaded from SQL database.')
                return True
            except Exception as e:
                print(f'Failed to load study from SQL database: {e}')
                return False
        else:
            print('Invalid load type or missing database URL.')
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


    def callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):

        if self.load_type == 'file':
            with open(f'{self.output_path}/{self.config.model_name}.pkl', 'wb') as f:
                pickle.dump(study, f)
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if self.load_type == 'file':
                self.create_log()

            if study.best_trial.number == trial.number:
                # saving best model
                print(f'Saving model for trial {trial.number}')
                self.export_model()

        

    def run_trial(self, trial):
        # deepcopy so that we do not accidentally modify base config
        config = copy.deepcopy(self.config)
        
        # Now we modify the hyper params in the config with Optuna
        # These are passed through the trial parameter
        config.optimizer = trial.suggest_categorical("optimizer", ["Adam"])
        config.activation_function = trial.suggest_categorical("activation_function", ["sigmoid"])

        config.learning_rate = trial.suggest_float('lr', config.lr_min, config.lr_max, log=True)

        # we set the loss function and parameters, and initialize the loss function in the config directly
        # ugly code but w.e
        # note the correct config name this time :)
        lossfn = config.loss_function
        
        config.loss_function = self.trial_builder.suggest_loss_params(trial, lossfn)
        
        # run trial
        out = self.runner.run(config,trial)

        trial.set_user_attr("metrics", out["metrics"])
        # print(f'\n\n out: {out} \n\n')
        # print(f'\n\n MAE: {out["mae"]} \n\n')

        # save temp model
        self.temp_model = out["model"]

        # return float to optuna optimizer call
        return out["mae"]
    
    def export_model(self):
        # train and save model based on a config
        torch.save(self.temp_model, f"{self.output_path}/{self.config.model_name}.pth")


    def create_log(self):
        export_logs(self.study, self.output_path, self.config.model_name)