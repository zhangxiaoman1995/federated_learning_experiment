import copy
import numpy as np
import torch
import math

from torch import nn
from torch.utils.data import DataLoader, Dataset
from options import args_parser
from load_data import DatasetSplit
from test import test_img
from optimizer.scaffold import SCAFFOLDOptimizer

class ScaffoldLocalUpdate:
    def __init__(self, args, traindata, testdata, net, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.net = net
        self.traindata = traindata
        self.testdata = testdata
        self.ldr_train = DataLoader(DatasetSplit(self.traindata, idxs), batch_size=self.args.local_bs, shuffle=True)
        ''' 
        layers = [self.net.conv1, self.net.conv2, self.net.fc1, self.net.fc2, self.net.fc3]
        weights = [{'params': layer.weight} for layer in layers]
        biases = [{'params': layer.bias, 'lr': 2 * self.args.lr} for layer in layers]
        param_groups = [None] * (len(weights) + len(biases))
        param_groups[::2] = weights
        param_groups[1::2] = biases
        self.optimizer = SCAFFOLDOptimizer(param_groups, lr=self.args.lr, weight_decay=0.004)
        '''
        self.optimizer = SCAFFOLDOptimizer(self.net.parameters(), lr=self.args.lr, weight_decay=0.004)

        self.controls = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        self.server_controls = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        self.delta_controls = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        self.delta_model = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        self.server_model = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        self.local_model = copy.deepcopy(list(self.net.parameters()))
        self.server_grad = copy.deepcopy(list(self.net.parameters()))
        self.pre_local_grad = copy.deepcopy(list(self.net.parameters()))

    def set_parameters(self, server_model):
        for old_param, new_param, local_param, server_param in zip(self.net.parameters(), server_model.parameters(), self.local_model, self.server_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
            server_param.data = new_param.data.clone()
            if(new_param.grad != None):
                if(old_param.grad == None):
                    old_param.grad = torch.zeros_like(new_param.grad)

                if(local_param.grad == None):
                    local_param.grad = torch.zeros_like(new_param.grad)

                old_param.grad.data = new_param.grad.data.clone()
                local_param.grad.data = new_param.grad.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for net_grad, new_grad in zip(self.net.parameters(), new_grads):
                net_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, net_grad in enumerate(self.net.parameters()):
                net_grad.data = new_grads[idx]

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
            if(param.grad != None):
                if(clone_param.grad == None):
                    clone_param.grad = torch.zeros_like(param.grad)
                clone_param.grad.data = param.grad.data.clone()

        return clone_param

    def get_grads(self, grads):
        self.optimizer.zero_grad()
        
        for x, y in DataLoader(self.ldr_train, len(self.ldr_train)):
            x, y = x.to(self.args.device), y.to(self.args.device)
            output = self.net(x)
            loss = self.loss_func(output, y)
            loss.backward()
        self.clone_model_paramenter(self.net.parameters(), grads)
        return grads
    
    def train(self):
        #self.net.train()
        #grads = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        #self.get_grads(grads)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            self.net.train()
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                self.optimizer.step(self.server_controls, self.controls)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        acc_test, loss_test = test_img(self.net, self.testdata, self.args)
        # get loss and acc after an train of local round

        # get net difference
        for local, server, delta in zip(self.net.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get client new controls
        new_controls = [torch.zeros_like(p.data) for p in self.net.parameters() if p.requires_grad]
        for server_control, control, new_control, delta in zip(self.server_controls, self.controls, new_controls,
                                                                self.delta_model):
            #import pdb;pdb.set_trace()
            #a = 1 / (math.ceil(len(self.ldr_train) / self.args.local_bs))
            a = 0.0001
            new_control.data = control.data - server_control.data - delta.data * a

        # get controls differences
        for control, new_control, delta in zip(self.controls, new_controls, self.delta_controls):
            delta.data = new_control.data - control.data
            control.data = new_control.data
        
        return acc_test, sum(epoch_loss) / len(epoch_loss)
'''
    def get_params_norm(self):
        params = []
        controls = []

        for delta in self.delta_model:
            params.append(torch.flatten(delta.data))

        for delta in self.delta_controls:
            controls.append(torch.flatten(delta.data))

        # return torch.linalg.norm(torch.cat(params), 2)
        return float(torch.norm(torch.cat(params))), float(torch.norm(torch.cat(controls)))
'''
