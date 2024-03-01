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
    num_trials = ht.config.num_trials
    print(f'\n\n****** Running {num_trials} trials ******\n\n')

    study = ht.get_study()

    # some params to improve the search efficiency perhaps ? :)
    # defaults look ok
    # because of the queued trial, n_trials should be the number you want to run +1
    study.optimize(ht.run_trial, n_trials=num_trials, callbacks=[ht.callback])

  
if __name__ == "__main__":
    main()