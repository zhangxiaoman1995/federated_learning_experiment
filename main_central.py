from models import LeNet
import numpy as np
import torch
from options import args_parser
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from test import test_img
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    batch_size = 256
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    test_dataset = MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = LeNet().to(args.device)
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss()
    epoch = 100

    writer = SummaryWriter('./runs/t_centerlize')
    for _epoch in range(epoch):
        epoch_loss = []
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x, train_label = train_x.to(args.device), train_label.to(args.device)
            #label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            _error.backward()
            sgd.step()
            epoch_loss.append(_error)
        avg_epoch = sum(epoch_loss) / len(epoch_loss)
        writer.add_scalar("train_loss", avg_epoch, _epoch)
        print('Round {:3d}, Average loss {:.3f}'.format(_epoch, avg_epoch))
        
    acc_test, loss_test = test_img(model, test_dataset, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
