from enum import unique
from platform import node
from tkinter.tix import Select
from numpy import dtype, einsum
import torch
from torch import embedding, nn, rand
import torch.nn.functional as F
from bisect import bisect_left
from scipy.sparse import csr_matrix, block_diag, coo_matrix
import numpy as np
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import math
from tqdm import tqdm
from datetime import datetime, timedelta

class T_RED_GNN(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN, self).__init__()
        self.n_rel = len(params.relation_vocab)
        self.device = params.device
        self.n_ent = len(params.entity_vocab)
        self.n_time = len(params.time_vocab)
        self.dataset = params.graph
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_layer = params.n_layer
        self.rela_embed_layer = nn.ModuleList([nn.Embedding(self.n_rel+1, self.hidden_dim) for _ in range(self.n_layer)])
        self.attention_1_layer = nn.ModuleList([nn.Linear(self.hidden_dim * 3, self.attn_dim, bias=False) for _ in range(self.n_layer)])
        self.attention_2_layer = nn.ModuleList([nn.Linear(self.attn_dim, 1, bias=False) for _ in range(self.n_layer)])
        self.linear_classifier = nn.Linear(self.hidden_dim, 1)

        self.past_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.now_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.future_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.time_embed = nn.Embedding(self.n_time, self.hidden_dim)
        if params.act == 'tanh':
            self.act = torch.tanh
        elif params.act == 'sigmoid':
            self.act = torch.sigmoid
        elif params.act == 'relu':
            self.act = torch.relu
        elif params.act == 'idd':
            self.act = lambda x : x
        elif params.act == 'softplus':
            self.act = F.softplus
        elif params.act == 'glu':
            self.act = F.glu

        self.dropout = nn.Dropout(params.dropout)
        self.init_param()

        with open('id2entity.pickle', 'rb') as handle:
            self.id2entity = pickle.load(handle)
        
        with open('id2relation.pickle', 'rb') as handle:
            self.id2relation = pickle.load(handle)

        with open('id2time.pickle', 'rb') as handle:
            self.id2time = pickle.load(handle)
        
        self.attention_vis = {'t1': {}, 't2': {}}
    
    def get_data(self, id, type):
        if type == 'entity':
            return self.id2entity[id]
        elif type == 'relation':
            return self.id2relation[id]
        elif type == 'time':
            return self.id2time[id]

        # self.attention_vis = []
        # self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        # cover rate
        # self.sp2o = {}
        # for spot in self.dataset:
        #     sp = tuple(spot[:2])
        #     if sp not in self.sp2o.keys():
        #         self.sp2o[sp] = [spot[2]]
        #     else:
        #         self.sp2o[sp].append(spot[2])
    def init_param(self):
        # init for self.rela_embed_layer
        for i in range(self.n_layer):
            nn.init.xavier_uniform_(self.rela_embed_layer[i].weight)
        nn.init.xavier_uniform_(self.time_embed.weight)

    # combine some time intervals
    def forward(self, batch, mode='train'):
        batch_size = batch['head'].shape[0]
        query_rel = batch['relation']
        entity = batch['head']
        query_time = batch['time']
        if mode == 'train':
            dataset = np.delete(self.dataset, batch['example_idx'], axis=0)
        else:
            dataset = self.dataset
        cur_entity = torch.column_stack([torch.arange(batch_size, device=self.device), entity])
        hidden_embedding = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        # h0 = torch.zeros((1, batch_size, self.hidden_dim), device=self.device)

        current_time = datetime(2014, 1, 1) + timedelta(days=int(query_time.cpu().numpy().tolist()[0]))
        if current_time <= datetime(2014, 6, 30):
            target_key = 't1'
        else:
            target_key = 't2'

        query_rel_id = query_rel.cpu().numpy().tolist()[0]
        if query_rel_id not in self.attention_vis[target_key].keys():
            self.attention_vis[target_key][query_rel_id] = np.zeros((self.n_rel, 2))

        adj_batch = coo_matrix(
                (
                    np.ones(dataset.shape[0]), 
                    (np.arange(dataset.shape[0]), dataset[:,0])
                ), shape=(dataset.shape[0], self.n_ent)
            )
        # Get new index
        # node_new_index = coo_matrix((np.arange(1, cur_entity.shape[0]+1), (cur_entity[:, 0], cur_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.int64).toarray()
        node_new_index = torch.sparse_coo_tensor(
            indices=cur_entity.T,
            values=torch.arange(1, cur_entity.shape[0]+1, device=self.device),
            size=(batch_size, self.n_ent),
            device=self.device,
            requires_grad=False
        ).to_dense()
        
        for i in range(self.n_layer):
            # Get neighbors

            # current entity to one-hot vector, also grounp by batch index as a vector
            batch_data_sp_vec = coo_matrix((np.ones(cur_entity.shape[0]), (cur_entity[:,1].tolist(), cur_entity[:,0].tolist())),shape=(self.n_ent, batch_size))
            select_one_hot = adj_batch @ batch_data_sp_vec
            select_idx = np.nonzero(select_one_hot)
            select_relation = np.concatenate([select_idx[1].reshape(-1, 1), dataset[select_idx[0], :]], axis=1)
            select_relation = torch.from_numpy(select_relation).to(self.device)

            # random_select_relation = np.random.randint(select_relation.shape[0], size=int(0.5 * select_relation.shape[0]))
            # select_relation = select_relation[random_select_relation, :]
            relative_time_index = select_relation[:, 4] - query_time[select_relation[:, 0]]
            # relative_time_index[torch.abs(relative_time_index) > 366] = 367

            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed_layer[i](select_relation[:,2]) + self.time_embed(torch.abs(relative_time_index))

            transformed_embed = torch.zeros_like(embed_rel, device=self.device)
            transformed_embed[relative_time_index > 0] = self.future_linear(embed_rel[relative_time_index > 0])
            transformed_embed[relative_time_index == 0] = self.now_linear(embed_rel[relative_time_index == 0])
            transformed_embed[relative_time_index < 0] = self.past_linear(embed_rel[relative_time_index < 0])

            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed_layer[i](select_relation[:,2]), self.rela_embed_layer[i](query_rel[select_relation[:, 0]])], dim=1)
            attention_score = torch.sigmoid(self.attention_2_layer[i](F.relu(self.attention_1_layer[i](attention_input))))
            msg_pass = attention_score * transformed_embed

            query_rel_set = set(select_relation[:, 2].cpu().numpy().tolist())
            for idx, target_rel in enumerate(query_rel_set):
                self.attention_vis[target_key][query_rel_id][target_rel][0] = self.attention_vis[target_key][query_rel_id][target_rel][0] + attention_score[select_relation[:, 2] == target_rel].sum().item()
                self.attention_vis[target_key][query_rel_id][target_rel][1] = self.attention_vis[target_key][query_rel_id][target_rel][1] + (select_relation[:, 2] == target_rel).sum().item()

            # attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed(select_relation[:,2]), self.rela_embed(query_rel[select_relation[:, 0]])], dim=1)
            # attention_score = self.attention_2(F.relu(self.attention_1(attention_input)))


            # get unique (batch_idx, entity) as neighbours,
            # node_new_index: from entity to new index (how are embeddings listed)
            # new_entity = np.unique(select_relation[:, [0, 3]], axis=0)
            new_entity = self.unique(select_relation[:, [0, 3]])
            # node_new_index = coo_matrix((np.arange(1, new_entity.shape[0]+1), (new_entity[:, 0], new_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.int64).toarray()

            node_new_index = torch.sparse_coo_tensor(
                indices=new_entity.T,
                values=torch.arange(1, new_entity.shape[0]+1, device=self.device),
                size=(batch_size, self.n_ent),
                device=self.device,
                requires_grad=False
            ).to_dense()

            
            new_index = node_new_index[select_relation[:,0], select_relation[:,3]]-1

            # attention_score = scatter_softmax(attention_score, new_index, dim=0)
            # msg_pass = attention_score * transformed_embed
            
            hidden_embedding = scatter(msg_pass, new_index, dim=0, reduce='sum')

            # h0 = torch.zeros((1, new_entity.shape[0], hidden_embedding.shape[1])).index_copy_(1, node_new_index[cur_entity[:, 0], cur_entity[:, 1]]-1, h0)
            # hidden_embedding = torch.tanh(hidden_embedding)
            hidden_embedding = self.act(self.dropout(hidden_embedding))
            # hidden_embedding = self.act(hidden_embedding)
            # hidden_embedding, h0 = self.gate(hidden_embedding.unsqueeze(0), h0)

            # row_col = np.array([node_new_index[select_relation[:,0], select_relation[:,3]]-1, np.arange(select_relation.shape[0])])
            # sparse_matrix = torch.sparse_coo_tensor(row_col, attention_score.squeeze(), size=(new_entity.shape[0], select_relation.shape[0]), device=self.device)
            # sparse_matrix = torch.sparse_coo_tensor(row_col, torch.ones(select_relation.shape[0]), size=(new_entity.shape[0], select_relation.shape[0]), device=self.device)
            # hidden_embedding = torch.sparse.mm(sparse_matrix, msg_pass)

            cur_entity = new_entity
            # print('ok')
        
        # calculate the cover rate
        # print(f'Coverage rate is {100 * self.coverage_rate(batch, cur_entity)}%')
        result = self.linear_classifier(hidden_embedding).reshape(-1)
        score_all = torch.zeros(batch_size, self.n_ent, device=self.device)
        score_all[cur_entity[:, 0], cur_entity[:, 1]] = result
        return score_all

    def unique(self, A):
        return torch.sparse_coo_tensor(
            indices=A.T,
            values=torch.ones(A.shape[0]),
            device=self.device,
            requires_grad=False
        ).coalesce().indices().T
        # coordinate = torch.nonzero(Sp)
        # return coordinate

    # def coverage_rate(self, batch, cur_entity):
    #     batch_size = batch['head'].shape[0]
    #     coverage_rate = []
    #     for b_idx in range(batch_size):
    #         o_entity_idx = np.where(cur_entity[:,0] == b_idx)
    #         o_entity = cur_entity[o_entity_idx, 1][0].tolist()
    #         sp2o_b = self.sp2o[(batch['head'][b_idx].item(), batch['relation'][b_idx].item())]
    #         inter_length = len(set(o_entity) & set(sp2o_b))
    #         coverage_rate.append(1.0 * inter_length / len(sp2o_b))
    #     return np.mean(coverage_rate)
