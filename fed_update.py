import copy
import torch
from torch import nn

def cal_ave_weight(w):
    w_avg = copy.deepcopy(w[0])
    #import pdb; pdb.set_trace()
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
