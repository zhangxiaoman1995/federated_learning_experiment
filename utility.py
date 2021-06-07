import os
import copy
import torch
import numpy as np
from torch import nn
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt

def gen_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def cal_ave_weight(w):
    #w_avg = torch.zeros_like(w[0])
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def save_model(self):
    model_path = os.path.join("models", self.dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(self.model, os.path.join(model_path, "server" + ".pt"))


def load_model(self):
    model_path = os.path.join("models", self.dataset, "server" + ".pt")
    assert (os.path.exists(model_path))
    self.model = torch.load(model_path)


def save_acc_loss(value, path, iter, savename):
    writer = SummaryWriter(gen_path(path))
    writer.add_scalar(savename, value, iter)
    writer.close()


def cal_ave_weight(w):
    w_avg = copy.deepcopy(w[0])
    #import pdb; pdb.set_trace()
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def plot_acc_loss(npy_folder):

    x_axis = [i for i in range(400)]
    m1 = list(np.load(npy_folder + 'niid_fedprox_lr0.01_simi0.5_400epoch.npy'))
    m2 = list(np.load(npy_folder + 'niid_scaffold_lr0.001_simi0.0_400epoch.npy'))
    m3 = list(np.load(npy_folder + 'niid_scaffold_lr0.01_simi0.5_400epoch.npy'))
    m4 = list(np.load(npy_folder + 'niid_fedprox_lr0.01_simi0.0_400epoch.npy'))
    m5 = list(np.load(npy_folder + 'niid_scaffold_lr0.01_simi0.0_400epoch.npy'))
    #m6 = np.load('npy_folder' + save_list[i])


    #sub_axix = filter(lambda x:x%200 == 0, x_axix)
    plt.title('Result Analysis')
    plt.figure(figsize=(20,10))
    plt.plot(x_axis, m1, color='green', label='fedprox_lr0.01_simi0.5', linestyle="-")
    plt.plot(x_axis, m2, color='red', label='scaffold_lr0.001_simi0.0', linestyle="-.")
    plt.plot(x_axis, m3,  color='blue', label='scaffold_lr0.01_simi0.5', linestyle=":")
    plt.plot(x_axis, m4, color='skyblue', label='fedprox_lr0.01_simi0.0', linestyle="--")
    plt.plot(x_axis, m5, color='coral', label='scaffold_lr0.01_simi0.0', linestyle="-")
    plt.legend() # 显示图例

    plt.xlabel('iteration times')
    #plt.yscale('log')
    plt.ylabel('Test Acc')
    plt.savefig('acc.png')
    #plt.show()

plot_acc_loss('acc_npy/')
