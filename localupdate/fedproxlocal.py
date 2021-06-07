import copy
import numpy as np
import torch

from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from options import args_parser
from load_data import DatasetSplit
from models import LeNet
from utility import *
from test import test_img
from optimizer.fedprox import FedProxOptimizer

class FedProxLocalUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        #self.optimizer = FedProxOptimizer(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, nesterov = False, weight_decay=0, mu=self.args.mu)

    def train(self, net):
        self.optimizer = FedProxOptimizer(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, nesterov = False, weight_decay=0, mu=self.args.mu)
        net.train()
        # train and update
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
