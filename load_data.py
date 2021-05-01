import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import random
from optimizer.fedprox import FedProx
from optimizer.scaffold import Scaffold

def random_split_iid(dataset, num_nodes):
    num_node = int(len(dataset)/num_nodes)
    dict_nodes, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_nodes):
        dict_nodes[i] = set(np.random.choice(all_idxs, num_node, replace=False))
        all_idxs = list(set(all_idxs) - dict_nodes[i])
    return dict_nodes

'''
def split_noniid(dataset, num_nodes):
    datalabel = dataset.targets
    datalabel_list = datalabel.tolist()
    num_node = int(len(dataset)/num_nodes)
    dict_nodes = {}
    for i in range(num_nodes):
        dict_nodes[i] = set([index for (index,value) in enumerate(datalabel_list) if value == i])
        #all_idxs = list(set(all_idxs) - dict_nodes[i])
    return dict_nodes
'''

def split_noniid_shuffle(dataset, num_nodes, similarity_percentage=args.similarity):
    datalabel = dataset.targets
    datalabel_list = datalabel.tolist()
    dict_nodes, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_nodes):
        index_i= [index for (index,value) in enumerate(datalabel_list) if value == i]
        index_non_iid = index_i[0: int((1-similarity_percentage)*len(index_i))]
        dict_nodes[i] = set(index_non_iid)

        all_idxs = list(set(all_idxs) - set(index_non_iid))
    
    if similarity_percentage == 0:
        continue
    else:
        num_node = int(len(all_idxs)/num_nodes)

        for i in range(num_nodes):
            value = dict_nodes[i]
            dict_choice = set(np.random.choice(all_idxs, num_node, replace=False))
            dict_nodes[i] = value | dict_choice
            all_idxs = list(set(all_idxs) - set(dict_choice))

    return dict_nodes

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        
        if self.args.optimizer ==  'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if self.args.optimizer == 'FedProx':
            optimizer = FedProx(net.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    nesterov = False,
                                    weight_decay=0,
                                    mu=self.args.mu)
        if self.args.optimizer == 'Scaffold':
            optimizer = Scaffold(net.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    nesterov = False,
                                    weight_decay=0,
                                    eta=self.args.eta,
                                    global_grad_data=self.args.global_grad_data)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class CentralUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

