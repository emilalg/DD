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
import numpy as np

class Runner:

    config = Config()
    trial_params = None 
    
    data = None
    DEVICE: torch.device

    def __init__(self, config: Config):   
        self.config = config
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
    
    def run(self, config: Config, trial=None):

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
            # check if should prune the trial using the median of the last 3 fscores
            # also prune if nan values exist
            if trial is not None:
                val_loss = valid_logs[i]['fscore']
                print(f'val_loss: {val_loss}')
                trial.report(val_loss, i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # scuffed nan check
                last_t_log = train_logs[-1]
                last_v_log = valid_logs[-1]

                t_contains_nan = np.any(np.isnan(list(last_t_log.values())))
                v_contains_nan = np.any(np.isnan(list(last_v_log.values())))

                if t_contains_nan or v_contains_nan:
                    print('Pruning nan.')
                    raise optuna.TrialPruned()
        
        # run predictions
        mae = None

        metrics = copy.deepcopy(valid_logs).pop()

        # :)
        if self.DEVICE.type != 'cuda':
            print(f'Warning: Using l1 loss instead of MAE.')
            vlc = copy.deepcopy(valid_logs)
            mae = vlc.pop()["l1_loss"]
        else:
            mae = process_testsubmission_mode(self.predictions_dataloader, model, self.ground_truths)
            metrics["mae"] = mae

        print(f'\n Mean Absolute Error: {mae} \n')

        

        out = {
            "train_logs": train_logs,
            "valid_logs": valid_logs,
            "model" : model,
            "mae": mae,
            "model": model,
            "metrics": metrics
        }
        return out
        