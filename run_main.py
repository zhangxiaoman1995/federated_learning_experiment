import copy
import numpy as np
import torch
import sys

from numpy import *
from torchvision import datasets, transforms
from torch import nn, autograd
from torchsummary import summary
from torchvision.datasets import MNIST
from load_data import random_split_iid, split_noniid_shuffle
from options import args_parser
from models import LeNet
import serverupdate
from serverupdate.central import CentralServer
from serverupdate.scaffold import ScaffoldSever
from serverupdate.fedprox import FedProxServer
from serverupdate.fedavg import FedAvgServer
from test import test_img


def main():
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    #set seed for device
    torch.cuda.manual_seed(args.seed)

    #save loss and acc
    loss_path = './runs/'+args.save_path
    acc_path = './test_acc/'+args.save_path


    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

    # sample users
    if args.distribution == 'iid':
        dict_users = random_split_iid(dataset_train, args.num_nodes)
    if args.distribution == 'non_iid':
        dict_users = split_noniid_shuffle(dataset_train, args.num_nodes, args.similarity)
    sum = 0
    for i in range(args.num_nodes):
        sum+=len(dict_users[i])
    print(sum)

    #import pdb;pdb.set_trace()
    net = LeNet().to(args.device)

    #print(summary(net,(1,28,28)))

    if args.optimizer=='centralized':
        update = CentralServer(args, dataset_train, dataset_test, net, dict_users)

    if args.optimizer=='fedprox':
        update = FedProxServer(args, dataset_train, dataset_test, net, dict_users)
    
    if args.optimizer=='scaffold':
        update = ScaffoldSever(args, dataset_train, dataset_test, net, dict_users)

    if args.optimizer=='fedavg':
        update = FedAvgServer(args, dataset_train, dataset_test, net, dict_users)

    update.train()

if __name__ == '__main__':
    main()
