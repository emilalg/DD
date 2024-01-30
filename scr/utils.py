import os

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
    #Read data from text file
    data = np.loadtxt('MG_DDSM/logs/unet.txt')

    #Split into variables, for plotting.
    epoch = data[:, 0]
    train_accuracy = data[:, 6] * 100
    valid_accuracy = data[:, 14] * 100

    #Rest of this is just standard plotting code.
    plt.plot(epoch, train_accuracy, label='Training Accuracy')
    plt.plot(epoch, valid_accuracy, label='Validation Accuracy')

    plt.ylim(90, 100)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


from dotenv import load_dotenv

"""
Legacy code for splitting training data. (Potential for repurpose?)
"""
import splitfolders

def load_env():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_directory, "..", ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)
