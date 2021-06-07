#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch


import localupdate
from localupdate.fedproxlocal import FedProxLocalUpdate
from torch.utils.tensorboard import SummaryWriter
from load_data import random_split_iid, split_noniid_shuffle
from options import args_parser
from models import LeNet
from utility import *
from test import test_img

class FedProxServer(FedProxLocalUpdate):
    def __init__(self, args, traindata, testdata, net, dict_users):
        #super().__init__(traindata, testdata, net[0], dict_users)

        self.users = []
        self.dict_users = dict_users
        self.net = net
        self.args = args
        self.testdata = testdata
        for idx in range(args.num_nodes):
            user = FedProxLocalUpdate(args, traindata, dict_users[idx])
            self.users.append(user)

    def train(self):

        loss_path = './runs/'+self.args.save_path
        acc_path = './test_acc/'+self.args.save_path

        writer_loss = SummaryWriter(gen_path(loss_path))
        writer_acc = SummaryWriter(gen_path(acc_path))

        self.net.train()
        # copy weights
        w_glob = self.net.state_dict()

        # training
        #loss_train = []
        
        w_locals = [w_glob for i in range(self.args.num_nodes)]
        
        all_loss = []
        all_acc = []

        for iter in range(self.args.epochs):
            loss_locals = []
            #local train 
            for idx, user in enumerate(self.users):
                w, loss = user.train(net=copy.deepcopy(self.net).to(self.args.device))
                w_locals[idx] = copy.deepcopy(w)
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = cal_ave_weight(w_locals)

            # copy weight to net
            self.net.load_state_dict(w_glob)
            # cal loss
            loss_avg = sum(loss_locals) / len(loss_locals)

            #test after a round
            #net.eval()
            acc_test, loss_test = test_img(self.net, self.testdata, self.args)

            all_loss.append(loss_avg)
            all_acc.append(acc_test)

            writer_acc.add_scalar("test_acc", acc_test, iter)
            writer_loss.add_scalar("train_loss", loss_avg, iter)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            print('Round {:3d}, Average test ACC {:.3f}'.format(iter, acc_test))


        all_loss = np.array(all_loss)
        np.save('loss_npy/' + self.args.save_lossdir +'.npy', all_loss)
        all_acc = np.array(all_acc)
        np.save('acc_npy/' + self.args.save_accdir +'.npy', all_acc)
