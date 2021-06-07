import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from options import args_parser
import random

args = args_parser()

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
    
    if similarity_percentage == 0.0:
        return dict_nodes

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
