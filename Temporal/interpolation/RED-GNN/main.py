from json import load
from logging.config import valid_ident
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from model import T_RED_GNN, T_RED_GNN_v2
from load_data import DataLoader
from utlis import cal_performance, cal_ranks
from old_utils import cal_mrr

def evaluate(model, loader):
    batch_size = 50

    n_data = loader.n_valid
    n_batch = n_data // batch_size + (n_data % batch_size > 0)
    ranking = []
    model.eval()
    ranking_old = []
    for i in range(n_batch):
        start = i*batch_size
        end = min(n_data, (i+1)*batch_size)
        batch_idx = np.arange(start, end)
        subs, rels, objs = loader.get_batch(batch_idx, data='valid')
        scores = model([subs, rels], loader.tKG).data.cpu().numpy()
        filters = []
        for i in range(len(subs)):
            filt = loader.filters[(subs[i], rels[i])]
            filt_1hot = np.zeros((loader.n_ent, ))
            filt_1hot[np.array(filt)] = 1
            filters.append(filt_1hot)
            
        filters = np.array(filters)
        ranks = cal_ranks(scores, objs, filters)
        ranking += ranks
        ranks_old = cal_mrr(scores, objs)
        ranking_old += ranks_old
    ranking = np.array(ranking)
    ranking_old = np.array(ranking_old)
    v_mrr, v_h1, v_h10 = cal_performance(ranking)
    v_mrr_old = np.mean(1 / ranking_old)


    n_data = loader.n_test
    n_batch = n_data // batch_size + (n_data % batch_size > 0)
    ranking = []
    ranking_old = []
    model.eval()
    for i in range(n_batch):
        start = i*batch_size
        end = min(n_data, (i+1)*batch_size)
        batch_idx = np.arange(start, end)
        subs, rels, objs = loader.get_batch(batch_idx, data='test')
        scores = model([subs, rels], loader.tKG).data.cpu().numpy()
        filters = []
        for i in range(len(subs)):
            filt = loader.filters[(subs[i], rels[i])]
            filt_1hot = np.zeros((loader.n_ent, ))
            filt_1hot[np.array(filt)] = 1
            filters.append(filt_1hot)
            
        filters = np.array(filters)
        ranks = cal_ranks(scores, objs, filters)
        ranking += ranks
        ranks_old = cal_mrr(scores, objs)
        ranking_old += ranks_old
    ranking = np.array(ranking)
    t_mrr, t_h1, t_h10 = cal_performance(ranking)
    ranking_old = np.array(ranking_old)
    t_mrr_old = np.mean(1 / ranking_old)

    out_str = '[VALID] MRR:%.4f old_MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f old_MRR:%.4f H@1:%.4f H@10:%.4f \n'%(v_mrr, v_mrr_old, v_h1, v_h10, t_mrr, t_mrr_old, t_h1, t_h10)
    return v_mrr, out_str

loader = DataLoader('data/nell')

device = 'cuda:3'
params = {
    'n_layer': 5,
    'hidden_dim': 20,
    'attn_dim': 10,
    'n_rel': loader.n_rel,
    'n_ent': loader.n_ent,
    'device': device,
    'interval': 15
}
opt_params = {
    'lr': 0.0003,
    # 'clip': np.inf,
    'batch': 20,
    'decay_rate': 0.994,
}

# contents = Data_v2(dataset='ICEWS14_forecasting', add_reverse_relation=True)
# adj_train = contents.get_adj(data_type='train')
# adj_valid = contents.get_adj(data_type='valid')
# for k in adj_valid.keys():
#     if k in adj_train.keys():
#         adj_valid[k].extend(adj_train[k])
#         adj_valid[k] = np.array(adj_valid[k])
# for k in adj_train.keys():
#     adj_train[k] = np.array(adj_train[k])
# adj_list = contents.get_adj_list()
writer = SummaryWriter()
model = T_RED_GNN(params).to(device)
optimizer = optim.Adam(model.parameters(), lr=opt_params['lr'], weight_decay=0e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-2)
# scheduler = ExponentialLR(optimizer, 0.1)
epoch_loss = 0
model.train()
cnt = 0
batch_size = opt_params['batch']
for epoch in range(10):
    n_batch = loader.n_train // batch_size + (loader.n_train % batch_size > 0)
    model.train()

    for batch_ndx in tqdm(range(n_batch)):
        start = batch_ndx * batch_size
        end = min(loader.n_train, (batch_ndx+1)*batch_size)
        batch_idx = np.arange(start, end)

        triple = loader.get_batch(batch_idx)
        optimizer.zero_grad()
        scores = model([triple[:, 0], triple[:, 1]], loader.KG)

        pos_scores = scores[[torch.arange(len(scores), device=device), torch.LongTensor(triple[:, 2]).to(device) ]]
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt_params['clip'])
        optimizer.step()

        # avoid NaN
        for p in model.parameters():
            X = p.data.clone()
            flag = X != X
            X[flag] = np.random.random()
            p.data.copy_(X)
        epoch_loss += loss.item()
        if loss.item() < 1e5:
            writer.add_scalar('Loss/train', loss.item(), cnt)
        cnt += 1

        # if batch_ndx % 100 == 0:
        #     print(loss.item())
        
        # if cnt % 5000 == 3:
        #     break
        # break
    valid_mrr, out_str = evaluate(model, loader)
    print(valid_mrr, out_str)
writer.close()
