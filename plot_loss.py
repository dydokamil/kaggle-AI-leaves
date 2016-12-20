# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

from Deep_Learning.kaggle.leaves.tools import NB_EPOCH

with open("/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/losses2.txt", "r") as myfile:
    all_data = myfile.read()

all_data_list = all_data.split('][')
all_data_list = [loss_acc.split(',') for loss_acc in all_data_list]
all_data_list[0][0] = all_data_list[0][0][1:]
all_data_list[-1][-1] = all_data_list[-1][-1][:-1]
all_data_list = np.array(all_data_list, dtype=np.float64)

loss = np.array(all_data_list[:, 0])
accuracy = np.array(all_data_list[:, 1])

x = np.arange(0, len(all_data_list)) * NB_EPOCH
x_smooth = np.linspace(x.min(), x.max(), 100)
accuracy_smooth = spline(x, accuracy, x_smooth)
loss_smooth = spline(x, loss, x_smooth)

plt.plot(x_smooth, loss_smooth, color='r')
plt.plot(x_smooth, accuracy_smooth, color='g')
plt.show()

pass
