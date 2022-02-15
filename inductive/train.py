import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/WN18RR_v1')
parser.add_argument('--seed', type=str, default=1234)

args = parser.parse_args()

class Options(object):
    pass


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    gpu = select_gpu()
    torch.cuda.set_device(gpu)
    print('gpu:', gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    if dataset == 'WN18RR_v1':
        opts.lr = 0.005
        opts.lamb = 0.0002
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.21
        opts.act = 'idd'
        opts.n_layer = 5
        opts.n_batch = 100
    elif dataset == 'fb237_v1':
        opts.lr = 0.0092
        opts.lamb = 0.0003
        opts.decay_rate = 0.994
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.dropout = 0.23
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20
    elif dataset == 'nell_v1':
        opts.lr = 0.0021
        opts.lamb = 0.000189
        opts.decay_rate = 0.9937
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2460
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10

    elif dataset == 'WN18RR_v2':
        opts.lr = 0.0016
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.02
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset == 'fb237_v2':
        opts.lr = 0.0077
        opts.lamb = 0.0002
        opts.decay_rate = 0.993
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.3
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10
    elif dataset == 'nell_v2':
        opts.lr = 0.0075
        opts.lamb = 0.000066
        opts.decay_rate = 0.9996
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2881
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 100

    elif dataset == 'WN18RR_v3':
        opts.lr = 0.0014
        opts.lamb = 0.000034
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.28
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset == 'fb237_v3':
        opts.lr = 0.0006
        opts.lamb = 0.000023
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.27
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20
    elif dataset == 'nell_v3':
        opts.lr = 0.0008
        opts.lamb = 0.0004
        opts.decay_rate = 0.995
        opts.hidden_dim = 16
        opts.attn_dim = 3
        opts.dropout = 0.06
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10

    elif dataset == 'WN18RR_v4':
        opts.lr = 0.006
        opts.lamb = 0.000132
        opts.decay_rate = 0.991
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.dropout = 0.11
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10
    elif dataset == 'fb237_v4':
        opts.lr = 0.0052
        opts.lamb = 0.000018
        opts.decay_rate = 0.999
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.07
        opts.act = 'idd'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset == 'nell_v4':
        opts.lr = 0.0005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20

    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)

    model = BaseModel(opts, loader)

    best_mrr = 0
    for epoch in range(50):
        mrr, out_str = model.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)

        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
    print(best_str)

