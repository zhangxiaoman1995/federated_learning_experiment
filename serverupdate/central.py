from models import LeNet
import numpy as np
import torch
import os

from torch import nn, autograd
from options import args_parser
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from test import test_img
from utility import *
from torch.utils.tensorboard import SummaryWriter


class CentralServer:
    def __init__(self, args, traindata, testdata, net, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.net = net
        self.testdata = testdata
        self.ldr_train = DataLoader(traindata, batch_size=self.args.local_bs, shuffle=True)
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    
    def train(self):

        loss_path = gen_path('./runs/'+self.args.save_path)
        acc_path = gen_path('./test_acc/'+self.args.save_path)

        writer_loss = SummaryWriter(loss_path)
        writer_acc = SummaryWriter(acc_path)

        all_loss = []
        all_acc = []

        for glob_iter in range(self.args.epochs):
            epoch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
            #for idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #label_np = np.zeros((labels.shape[0], 10))
                self.optimizer.zero_grad()
                predict_y = self.net(images.float())
                loss = self.loss_func(predict_y, labels.long())
                #if idx % 10 == 0:
                #    print('idx: {}, loss: {}'.format(idx, loss))
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss)
            loss_avg = sum(epoch_loss) / len(epoch_loss)
            acc_test, loss_test = test_img(self.net, self.testdata, self.args)
            
            writer_loss.add_scalar("train_loss", loss_avg, glob_iter)
            writer_acc.add_scalar("test_acc", acc_test, glob_iter)
            print('Round {:3d}, Average loss {:.3f}'.format(glob_iter, loss_avg))
            print('Round {:3d}, Average ACC {:.3f}'.format(glob_iter, acc_test))

            all_loss.append(loss_avg)
            all_acc.append(acc_test)
        
        
        writer_loss.close()
        writer_acc.close()

        all_loss = np.array(all_loss)
        np.save('loss_npy/' + self.args.save_lossdir +'.npy', all_loss)
        all_acc = np.array(all_acc)
        np.save('acc_npy/' + self.args.save_accdir +'.npy', all_acc)
