import torch
import numpy as np
import time

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import RED_GNN_induc
from utils import cal_ranks, cal_performance, PeakMemoryMeter

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = RED_GNN_induc(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

        # Peak memory (MB) tracked separately for training vs inference
        self._train_mem_peak = None
        self._infer_mem_peak = None
        self._last_train_mem = {'cpu_rss_peak_mb': 0.0, 'cuda_alloc_peak_mb': 0.0, 'cuda_reserved_peak_mb': 0.0}

    def train_batch(self,):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        t_time = time.time()
        self.model.train()
        mem_meter = PeakMemoryMeter(track_cpu=True, track_cuda=True)
        mem_meter.reset()
        for i in range(n_batch):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:,0], triple[:,1])

            pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(triple[:,2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
            loss.backward()
            self.optimizer.step()
            mem_meter.update()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        mem_meter.update()
        train_mem = mem_meter.summary_mb()
        self._last_train_mem = train_mem
        if (self._train_mem_peak is None) or (train_mem['cuda_reserved_peak_mb'] > self._train_mem_peak['cuda_reserved_peak_mb']):
            self._train_mem_peak = train_mem
        self.scheduler.step()
        self.t_time += time.time() - t_time

        valid_mrr, out_str = self.evaluate()
        return valid_mrr, out_str

    def evaluate(self, ):
        batch_size = self.n_batch

        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        mem_meter = PeakMemoryMeter(track_cpu=True, track_cuda=True)
        mem_meter.reset()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels).data.cpu().numpy()
            mem_meter.update()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.val_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
             
            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1, v_h10 = cal_performance(ranking)


        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, 'inductive').data.cpu().numpy()
            mem_meter.update()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.tst_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent_ind, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
             
            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time
        mem_meter.update()
        infer_mem = mem_meter.summary_mb()
        if (self._infer_mem_peak is None) or (infer_mem['cuda_reserved_peak_mb'] > self._infer_mem_peak['cuda_reserved_peak_mb']):
            self._infer_mem_peak = infer_mem

        tm = self._last_train_mem
        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\t[PEAK_MEM_MB] train_cpu:%.1f train_cuda_alloc:%.1f train_cuda_rsv:%.1f inf_cpu:%.1f inf_cuda_alloc:%.1f inf_cuda_rsv:%.1f\n' %(
            v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time,
            tm['cpu_rss_peak_mb'], tm['cuda_alloc_peak_mb'], tm['cuda_reserved_peak_mb'],
            infer_mem['cpu_rss_peak_mb'], infer_mem['cuda_alloc_peak_mb'], infer_mem['cuda_reserved_peak_mb']
        )
        return v_mrr, out_str
