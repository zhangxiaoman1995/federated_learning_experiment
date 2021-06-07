import copy
import numpy as np
import torch
import os
import math

import localupdate
from localupdate.scaffoldlocal import ScaffoldLocalUpdate
from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from test import test_img
from utility import *

# Implementation for SCAFFOLD Server
class ScaffoldSever(ScaffoldLocalUpdate):
    def __init__(self, args, traindata, testdata, net, dict_users):
        super().__init__(args, traindata, testdata, net, dict_users)
        
        self.users = []
        self.dict_users = dict_users
        for idx in range(args.num_nodes):
            user = ScaffoldLocalUpdate(args, traindata, testdata, net, dict_users[idx])
            self.users.append(user)
            #self.total_train_samples += user.train_samples

        self.server_controls = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]

        print("Finished creating SCAFFOLD server.")

    def train(self):

        loss_path = gen_path('./runs/'+self.args.save_path)
        acc_path = gen_path('./test_acc/'+self.args.save_path)

        writer_loss = SummaryWriter(loss_path)
        writer_acc = SummaryWriter(acc_path)

        all_loss = []
        all_acc = []
        
        for glob_iter in range(self.args.epochs):
            #print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0

            self.send_parameters()
            # Evaluate model at each iteration
            #self.evaluate()
            loss = []
            #acc_test = []
            for user in self.users:
                user_acc, user_loss = user.train()
                #user.drop_lr()
                loss.append(user_loss)
            acc_test = user_acc
            #import pdb;pdb.set_trace()
            loss_avg = sum(loss) / len(loss)
            print('Round {:3d}, Average loss {:.3f}'.format(glob_iter, loss_avg))
            print('Round {:3d}, Average acc {:.3f}'.format(glob_iter, acc_test))
            all_loss.append(loss_avg)
            all_acc.append(acc_test)
            writer_loss.add_scalar("train_loss", loss_avg, glob_iter)
            writer_acc.add_scalar("test_acc", acc_test, glob_iter)
            self.aggregate_parameters()
            #self.get_max_norm()
            #self.test_img(testdata)
        writer_loss.close()
        writer_acc.close()

        all_loss = np.array(all_loss)
        np.save('loss_npy/' + self.args.save_lossdir +'.npy', all_loss)
        all_acc = np.array(all_acc)
        np.save('acc_npy/' + self.args.save_accdir +'.npy', all_acc)

        #print('Test ACC, {:.3f}'.format(sum(acc_test)/len(acc_test)))
        #self.save_model()

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.net)
            for control, new_control in zip(user.server_controls, self.server_controls):
                control.data = new_control.data

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        total_samples = 0
        for idx, user in enumerate(self.users):
            total_samples += len(self.dict_users[idx])
        for user in self.users:
            self.add_parameters(user, total_samples)

    def add_parameters(self, user, total_samples):
        num_of_users = len(self.users)
        for param, control, del_control, del_model in zip(self.net.parameters(), self.server_controls,
                                                          user.delta_controls, user.delta_model):
            param.data = param.data + del_model.data / num_of_users
            control.data = control.data + del_control.data / num_of_users
