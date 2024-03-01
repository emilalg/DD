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
from hypertuner.tuner import Tuner


def main():    
    # initalize hypertuner
    ht = Tuner()
    config = ht.config
    num_trials = ht.config.num_trials
    print(f'\n\n****** Running {num_trials} trials ******\n\n')
    # we can make a study deterministic by assigning a custom sampler with a set seed
    # does not feel necessary atm

    # pruner args:
    #     n_startup_trials:
    #         Pruning is disabled until the given number of trials finish in the same study.
    #     n_warmup_steps:
    #         Pruning is disabled until the trial exceeds the given number of step. Note that
    #         this feature assumes that ``step`` starts at zero.
    #     interval_steps:
    #         Interval in number of steps between the pruning checks, offset by the warmup steps.
    #         If no value has been reported at the time of a pruning check, that particular check
    #         will be postponed until a value is reported.
    study = ht.get_study()

    # some params to improve the search efficiency perhaps ? :)
    # defaults look ok
    # because of the queued trial, n_trials should be the number you want to run +1
    study.optimize(ht.run_trial, n_trials=num_trials, callbacks=[ht.callback])


    

    
if __name__ == "__main__":
    main()