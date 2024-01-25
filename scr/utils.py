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

data = np.loadtxt('MG_DDSM/logs/unet.txt')

epoch = data[:, 0]
train_accuracy = data[:, 6] * 100
valid_accuracy = data[:, 14] * 100

plt.plot(epoch, train_accuracy, label='Training Accuracy')
plt.plot(epoch, valid_accuracy, label='Validation Accuracy')

plt.ylim(90, 100)

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


######
######  spliting data into train test and valid
######

import splitfolders

splitfolders.ratio("CMMD_Data/", output="CMMD_Dataset",
    seed=1, ratio=(.70, .15, .15), group_prefix=None, move=False) # default values