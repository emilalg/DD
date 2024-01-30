import os

def makedirs(path):
    import os

    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)


######
######  Ploting logs
######

import numpy as np
import matplotlib.pyplot as plt


def plotting_logs(logs_path):
    data = np.loadtxt(logs_path)

    epoch = data[:, 0]
    train_accuracy = data[:, 6] * 100
    valid_accuracy = data[:, 14] * 100

    plt.plot(epoch, train_accuracy, label="Training Accuracy")
    plt.plot(epoch, valid_accuracy, label="Validation Accuracy")

    plt.ylim(90, 100)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


from dotenv import load_dotenv


def load_env():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_directory, "..", ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)
