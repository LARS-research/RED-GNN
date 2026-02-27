import argparse
import os
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import select_gpu


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

parser = argparse.ArgumentParser(description="Parser for KG GNN")
parser.add_argument('--data_path', type=str, default='../data/family/')
parser.add_argument('--seed', type=str, default=1234)


args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["MKL_NUM_THREADS"] = "10"
    os.environ["OPENBLAS_NUM_THREADS"] = "10"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10" 
    torch.set_num_threads(10)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    opts = Options
    opts.perf_file = 'results/' + dataset + '_perf.txt'

    gpu = select_gpu()
    torch.cuda.set_device(gpu)
    print(gpu)

    def run_model(params):
        loader = DataLoader(args.data_path)
        opts.n_ent = loader.n_ent
        opts.n_rel = loader.n_rel
        opts.data_path = args.data_path

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        opts.lr = 10**params['lr']
        opts.lamb = 10**params["lamb"]
        opts.decay_rate = params['decay_rate']
        opts.hidden_dim = params['hidden_dim']
        opts.init_dim = params['hidden_dim']
        opts.attn_dim = params['attn_dim']
        opts.dropout = params['dropout']
        opts.act = params['act']
        opts.n_layer = params['n_layer']
        opts.n_batch = params['n_batch']
        #opts.n_samples = params['n_samples']
        opts.n_samples = None
        if dataset == 'YAGO':
            opts.n_tbatch = 2
        else:
            opts.n_tbatch = 10

        #opts.lr = 0.0009
        #opts.lamb = 0.00008
        #opts.decay_rate = 0.9938
        #opts.hidden_dim = opts.init_dim = 48
        #opts.attn_dim = 3
        #opts.act = 'relu'
        #opts.dropout = 0.0391
        #opts.n_layer = 5
        #opts.n_batch = 20

        # UMLS
        #opts.lr = 0.0012
        #opts.decay_rate = 0.9917
        #opts.lamb = 0.000115
        #opts.hidden_dim = 48
        #opts.init_dim = 48
        #opts.attn_dim = 5
        #opts.n_layer = 4
        #opts.dropout = 0.0024
        #opts.act = 'relu'
        #opts.n_batch = 20

        # WN18RR
        #opts.lr = 0.0021
        #opts.decay_rate = 0.9962
        #opts.lamb = 0.000037
        #opts.hidden_dim = opts.init_dim = 48
        #opts.attn_dim = 5
        #opts.n_layer = 5
        #opts.dropout = 0.0067
        #opts.act = 'tanh'
        #opts.n_batch = 100

        # fb15k-237
        #opts.lr = 0.0009
        #opts.decay_rate = 0.9938
        #opts.lamb = 0.000080
        #opts.hidden_dim = opts.init_dim = 48
        #opts.attn_dim = 3
        #opts.n_layer = 3
        #opts.dropout = 0.0391
        #opts.act = 'relu'
        #opts.n_batch = 20

        # nell
        #opts.lr = 0.0011
        #opts.decay_rate = 0.9938
        #opts.lamb = 0.000089
        #opts.hidden_dim = opts.init_dim = 48
        #opts.attn_dim = 5
        #opts.n_layer = 5
        #opts.dropout = 0.2593
        #opts.act = 'relu'
        #opts.n_batch = 5

        # fb237
        #opts.lr = 0.0009
        #opts.decay_rate = 0.9938
        #opts.lamb = 0.00008
        #opts.hidden_dim = opts.init_dim = 48
        #opts.attn_dim = 5
        #opts.n_layer = 4
        #opts.dropout = 0.0391
        #opts.act = 'relu'
        #opts.n_batch = 5
        #opts.n_samples = 20

        #opts.lr = 0.0001
        #opts.decay_rate = 0.9936
        #opts.lamb = 0.000082
        #opts.hidden_dim = opts.init_dim = 64
        #opts.attn_dim = 5
        #opts.n_layer = 3
        #opts.dropout = 0.1184
        #opts.act = 'relu'
        #opts.n_batch = 10
        #opts.n_samples = 30

        # yago
        #opts.lr = 0.0003
        #opts.decay_rate = 0.997
        #opts.lamb = 0.000111
        #opts.hidden_dim = opts.init_dim = 48
        #opts.attn_dim = 5
        #opts.n_layer = 3
        #opts.dropout = 0.2131
        #opts.act = 'relu'
        #opts.n_batch = 3

        try:
            config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %d, %.4f,%s, %s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.init_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act, str(opts.n_samples))
            print(config_str)

            model = BaseModel(opts, loader)

            best_mrr = 0
            early_stop = 0
            #MAX_STOP = 12
            for epoch in range(30):
            #    if early_stop > MAX_STOP:
            #        break
                mrr, out_str = model.train_batch()
                print(str(epoch) + '\t' + out_str)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_str = out_str
                    early_stop = 0
                else:
                    early_stop += 1
        except RuntimeError:
            best_mrr = 0
            print('run out of memory')
            return {'loss': -best_mrr, 'status': STATUS_OK}

        with open(opts.perf_file, 'a') as f:
            f.write(config_str)
            f.write(best_str + '\n')
            print('\n\n')

        return {'loss': -best_mrr, 'status': STATUS_OK}
    #loss = model.train_batch()
    #print(epoch, loss)


    space4kg = {
        'lr': hp.uniform('lr', -4, -2),
        'lamb': hp.uniform('lamb', -5, -3),
        'decay_rate': hp.uniform('decay_rate', 0.99, 1),
        'dropout': hp.uniform('dropout', 0, 0.3),
        'hidden_dim': hp.choice('hidden_dim', [16, 32, 64]),
        #'init_dim': hp.choice('init_dim', [5, 10, 25, 50]),
        'attn_dim': hp.choice('attn_dim', [5]),
        'act': hp.choice('act', ['idd', 'relu', 'tanh']),
        #'n_layer': hp.choice('n_layer', [1,2,3,4,5,6]),
        'n_layer': hp.choice('n_layer', [4,5]),
        #'n_batch': hp.choice('n_batch', [10,20,50,100])
        #'n_batch': hp.choice('n_batch', [5, 10, 20, 30, 50]),
        'n_batch': hp.choice('n_batch', [10, 20, 30]),
        #'n_samples': hp.choice('n_samples', [2, 5, 10, 15, 20]),
        }

    trials = Trials()
    best = fmin(run_model, space4kg, algo=partial(tpe.suggest, n_startup_jobs=20), max_evals=200, trials=trials)


