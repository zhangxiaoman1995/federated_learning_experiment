import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=400, help="rounds of training")
    parser.add_argument('--num_nodes', type=int, default=10, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--mu', type=float, default=0.01, help="mu parameter for fedprox")
    parser.add_argument('--eta', type=float, default=0.01, help="mu parameter for scaffold")
    parser.add_argument('--save_path', type=str, default='iid', help='trainning log save path')
    parser.add_argument('--optimizer', type=str, default= 'centralized', help='trainning log save path')
    parser.add_argument('--similarity', type=float, default=0.0, help="data distribution similarity for non_iid")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--distribution', type=str, default='non_iid', help='iid or non-iid split dataset')
    parser.add_argument('--save_lossdir', type=str, default='./runs', help='file dir to save loss')
    parser.add_argument('--save_accdir', type=str, default='./test_acc', help='file dir to save acc')

    args = parser.parse_args()
    return args
