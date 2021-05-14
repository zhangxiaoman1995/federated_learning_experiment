#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torchvision.datasets import MNIST
from load_data import random_split_iid, split_noniid, split_noniid_shuffle
from options import args_parser
from load_data import LocalUpdate
from models import LeNet
from fed_update import cal_ave_weight
from test import test_img
from torch.utils.tensorboard import SummaryWriter


def main():
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu))


    #set seed for device
    torch.cuda.manual_seed(args.seed)


    #save loss and acc
    loss_path = './runs/'+args.save_path
    acc_path = './test_acc/'+args.save_path
    
    def gen_path:(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    writer_loss = SummaryWriter(gen_path(loss_path))
    writer_acc = SummaryWriter(gen_path(acc_path))

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    # sample users
    if args.distribution == 'iid':
        dict_users = random_split_iid(dataset_train, args.num_nodes)
    if args.distribution == 'non_iid'
        dict_users = split_noniid(dataset_train, args.num_nodes, args.similarity)

    net_glob = LeNet().to(args.device)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    #loss_train = []
    
    w_locals = [w_glob for i in range(args.num_nodes)]
    
    if args.model == 'central':
        central = CentralUpdate(arg=args, dataset=dataset_train)
        loss = central.train()
        writer_loss.add_scalar("train_loss", loss_avg, iter)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        #test after a round
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        writer_acc.add_scalar("test_acc", acc_test, iter)

        writer_loss.close()
        
    else:
        for iter in range(args.epochs):
        loss_locals = []
            for idx in range(args.num_nodes):
                #import pdb; pdb.set_trace()
                local = LocalUpdate(args=args, dataset=dataset_train)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),idxs=idx)
                w_locals[idx] = copy.deepcopy(w)
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = cal_ave_weight(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            writer_loss.add_scalar("train_loss", loss_avg, iter)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

            #test after a round
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            writer_acc.add_scalar("test_acc", acc_test, iter)

        writer_loss.close()
    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

if __name__ == '__main__':
    main()
