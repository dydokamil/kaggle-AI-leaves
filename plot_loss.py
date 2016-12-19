# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from Deep_Learning.kaggle.leaves.tools import NB_EPOCH

with open("/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/losses.txt", "r") as myfile:
    all_data = myfile.read()

all_data_list = all_data.split('][')
all_data_list = [loss_acc.split(',') for loss_acc in all_data_list]
all_data_list[0][0] = all_data_list[0][0][1:]
all_data_list[-1][-1] = all_data_list[-1][-1][:-1]
all_data_list = np.array(all_data_list, dtype=np.float64)

losses = all_data_list[:, 0]
accuracy = all_data_list[:, 1]
x = np.arange(0, len(all_data_list)) * NB_EPOCH

# plt.plot(x, losses, color='r')
plt.plot(x, accuracy, color='g')
plt.show()

pass
