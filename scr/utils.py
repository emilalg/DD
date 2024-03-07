import argparse
import os
import re
from dataclasses import dataclass, fields
from dotenv import load_dotenv
import albumentations as A

"""
Function to fill in missing directories, given an input path.
Currently not in use anywhere, potential for repurpose?
:param path: The path to be filled in.
"""


def makedirs(path):
    import os

    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)


######
######  Ploting logs
######

"""
Template code for plotting training info, 
with added lines to read training data from text file.
Similar code can be found in train.py.
"""

import numpy as np
import matplotlib.pyplot as plt


def plotting_logs(logs_path):
    data = np.loadtxt(logs_path)

    epoch = data[:, 0]
    train_accuracy = data[:, 6] * 100
    valid_accuracy = data[:, 14] * 100
    # Read data from text file
    data = np.loadtxt("MG_DDSM/logs/unet.txt")

    # Split into variables, for plotting.
    epoch = data[:, 0]
    train_accuracy = data[:, 6] * 100
    valid_accuracy = data[:, 14] * 100

    # Rest of this is just standard plotting code.
    plt.plot(epoch, train_accuracy, label="Training Accuracy")
    plt.plot(epoch, valid_accuracy, label="Validation Accuracy")

    plt.ylim(90, 100)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_loss_key(loss_name):
    key_base = camel_to_snake(loss_name[:-4])
    return key_base + "_loss_weighted"


# Read the last line of the logs file
def read_log_results(logs_file_path):
    with open(logs_file_path, "r") as logs_file:
        lines = logs_file.readlines()  # Read all lines into a list
        return lines[-1]  # Return the last line


"""
Loading configuration, first from .env at project root, then from passed config arguments.
"""



@dataclass
class Config:
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    activation_function: str = "sigmoid"
    encoder: str = "resnet101"
    loss_function: str  = "DSCPlusPlusLoss"
    learning_rate: float = 0.0001 #0.0001
    learning_scheduler: str = "steplr"
    model_name: str = "double_d"
    num_epochs: int = 10 #5
    num_trials: int = 3
    num_workers: int = 4
    optimizer: str = "Adam"
    output_path: str = "test_output"
    prediction_data_path: str = "breast-density-prediction/test/test"
    prediction_mode: str = "testsubmission"
    pretrained_weights: str = None
    segmentation_model: str = "Unet"
    study_name: str = "Default"
    # To use the augmented images, change this path to breast-density-prediction-test/train/train
    train_data_path: str = "breast-density-prediction/train/train"
    #train_data_path: str = "breast-density-prediction-test/train/train"
    train_batch_size: int = 10 #10 #4 #16
    valid_batch_size: int = 4
    #Epochs # train batch size# Learning rate # BEst l1loss
    #10     # 10              # 0.0001        #  15.6
    #10     # 4               # 0.001         #

    #AdamW parameters
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0001

    #hypertuner
    lr_min: float = 0.0001
    lr_max: float = 0.0009
    alpha_min: float = 0.1
    alpha_max: float = 0.99
    beta_min: float = 0.1
    beta_max: float = 0.99
    gamma_min: float = 0.5
    gamma_max: float = 3.5
    pruning_interval: int = 6
    ce_ratio: float = 0.5
    smooth: float = 1.0

    def __post_init__(self):
        self = load_config_from_env(self)
        self = load_config_from_args(self)
        self.prediction_data_path = os.path.join(self.PROJECT_ROOT, self.prediction_data_path)
        self.train_data_path = os.path.join(self.PROJECT_ROOT, self.train_data_path)
        self.output_path = os.path.join(self.PROJECT_ROOT, self.output_path)
        
    @property
    def PROJECT_ROOT(self) -> str:
        return self._PROJECT_ROOT

    # prettier printing with newlines
    def __str__(self):
        attributes = [f"{name}={getattr(self, name)}" for name in self.__dict__]
        return "Config(\n" + ",\n".join(attributes) + "\n)"


def add_arguments_for_config(parser, config):
    for field in fields(Config):
        field_name = field.name
        field_type = field.type
        field_default = config.__getattribute__(field_name)
        if field_type == str:
            parser.add_argument(f"--{field_name}", default=field_default, type=str, help=f"{field_name} description")
        elif field_type == int:
            parser.add_argument(f"--{field_name}", default=field_default, type=int, help=f"{field_name} description")
        elif field_type == float:
            parser.add_argument(f"--{field_name}", default=field_default, type=float, help=f"{field_name} description")
        elif field_type == bool:
            parser.add_argument(f"--{field_name}", default=field_default, type=bool, help=f"{field_name} description")
        elif field_type == tuple:
            parser.add_argument(f"--{field_name}", default=field_default, type=tuple, help=f"{field_name} description")
        else:
            raise ValueError(f"Unsupported field type for {field_name}: {field_type}")


def load_config_from_env(config):
    if not config:
        config = Config()

    dotenv_path = os.path.join(config.PROJECT_ROOT, ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)

    for field in fields(Config):
        field_name = field.name
        field_type = field.type
        field_value = os.getenv(field_name, None)
        if field_value is not None:
            if field_type == str:
                setattr(config, field_name, field_value)
            elif field_type == int:
                setattr(config, field_name, int(field_value))
            elif field_type == float:
                setattr(config, field_name, float(field_value))
            elif field_type == bool:
                setattr(config, field_name, field_value.lower() == "true")
            elif field_type == tuple:
                if field_name == "betas":
                    # Parse the comma-separated values into a tuple of floats
                    setattr(config, field_name, tuple(map(float, field_value.split(","))))
                else:
                    setattr(config, field_name, tuple(map(field_type, field_value.split(","))))
            else:
                raise ValueError(f"Unsupported field type for {field_name}: {field_type}")

    return config


def load_config_from_args(config):
    parser = argparse.ArgumentParser()

    for field in fields(Config):
        field_name = field.name
        field_type = field.type
        field_default = config.__getattribute__(field_name)
        if field_type == str:
            parser.add_argument(f"--{field_name}", default=field_default, type=str, help=f"{field_name} description")
        elif field_type == int:
            parser.add_argument(f"--{field_name}", default=field_default, type=int, help=f"{field_name} description")
        elif field_type == float:
            parser.add_argument(f"--{field_name}", default=field_default, type=float, help=f"{field_name} description")
        elif field_type == bool:
            parser.add_argument(f"--{field_name}", default=field_default, type=bool, help=f"{field_name} description")
        elif field_type == tuple:
            parser.add_argument(f"--{field_name}", default=field_default, type=tuple, help=f"{field_name} description")
        else:
            raise ValueError(f"Unsupported field type for {field_name}: {field_type}")

    args = parser.parse_args()

    for field in fields(Config):
        field_name = field.name
        setattr(config, field_name, getattr(args, field_name))

    return config
