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

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


######
######  spliting data into train test and valid
######

"""
Legacy code for splitting training data. (Potential for repurpose?)
"""
import splitfolders

splitfolders.ratio("CMMD_Data/", output="CMMD_Dataset",
    seed=1, ratio=(.70, .15, .15), group_prefix=None, move=False) # default values