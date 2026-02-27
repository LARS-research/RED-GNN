from enum import unique
from platform import node
from numpy import dtype, einsum
import torch
from torch import embedding, nn, rand
import torch.nn.functional as F
from bisect import bisect_left
from scipy.sparse import csr_matrix, block_diag, coo_matrix
import numpy as np

class T_RED_GNN_temp(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN, self).__init__()
        self.n_layer = params['n_layer']
        self.hidden_dim = params['hidden_dim']
        self.attn_dim = params['attn_dim']
        self.n_rel = params['n_rel']
        self.device = params['device']
        self.n_ent = params['n_ent']
        self.rela_embed = nn.Embedding(2*self.n_rel+2, 20)
        self.attention_1 = nn.Linear(60, 30, bias=False)
        self.attention_2 = nn.Linear(30, 1, bias=False)
        self.linear_classifier = nn.Linear(20, 1)

    # combine some time intervals
    def forward(self, X, dataset):
        batch_size = X.rel_idx.shape[0]
        query_rel = X.rel_idx
        entity = X.src_idx
        target = X.target_idx
        for i in range(batch_size):
            query_data = np.array([entity[i], query_rel[i], target[i]])
            pre_idx = np.sum(dataset[entity[i]] == query_data, axis=1)
            pre_idx = np.nonzero(pre_idx != 3)[0]
            dataset[entity[i]] = dataset[entity[i]][pre_idx]
        cur_entity = np.array([[i, entity[i]] for i in range(batch_size)], dtype=np.int32)
        hidden_embedding = torch.zeros((batch_size, 20), device=self.device)

        # Get new index
        node_new_index = coo_matrix((np.arange(1, cur_entity.shape[0]+1), (cur_entity[:, 0], cur_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.longlong).toarray()
        
        for i in range(3):
            # Get neighbors
            # diagonal matrix group by batch index
            select_relation = [np.concatenate([b_idx * np.ones((dataset[ent].shape[0], 1), dtype=np.int32), dataset[ent]], axis=1) for b_idx, ent in cur_entity]
            select_relation = np.concatenate(select_relation, axis=0)
            random_idx = np.random.randint(select_relation.shape[0], size=int(0.8 * select_relation.shape[0]))
            select_relation = select_relation[random_idx, :]
            # embed_rel = hidden_embedding[node_new_index[select_relation[:,0]], :] + rel_embedding[select_relation[:,1], :]
            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed(torch.tensor(select_relation[:,2], device=self.device))


            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed(torch.tensor(select_relation[:,2], device=self.device)), self.rela_embed(torch.tensor(query_rel[select_relation[:, 0]], device=self.device))], dim=1)
            attention_score = torch.sigmoid(self.attention_2(F.relu(self.attention_1(attention_input))))

            # get unique (batch_idx, entity) as neighbours,
            # node_new_index: from entity to new index (how are embeddings listed)
            new_entity = np.unique(select_relation[:, [0, 3]], axis=0)
            node_new_index = coo_matrix((np.arange(1, new_entity.shape[0]+1), (new_entity[:, 0], new_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.longlong).toarray()


            row_col = np.array([node_new_index[select_relation[:,0], select_relation[:,3]]-1, np.arange(select_relation.shape[0])])

            sparse_matrix = torch.sparse_coo_tensor(row_col, attention_score.squeeze(), size=(new_entity.shape[0], select_relation.shape[0]), device=self.device)
            # sparse_matrix = torch.sparse_coo_tensor(row_col, torch.ones(select_relation.shape[0]), size=(new_entity.shape[0], select_relation.shape[0]))
            hidden_embedding = torch.sparse.mm(sparse_matrix, embed_rel)

            cur_entity = new_entity
            # print('ok')
        
        result = self.linear_classifier(hidden_embedding).reshape(-1)
        score_all = torch.zeros(batch_size, self.n_ent, device=self.device)
        score_all[cur_entity[:, 0], cur_entity[:, 1]] = result
        return score_all


class T_RED_GNN_v2(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN_v2, self).__init__()
        self.n_layer = params['n_layer']
        self.hidden_dim = params['hidden_dim']
        self.attn_dim = params['attn_dim']
        self.n_rel = params['n_rel']
        self.device = params['device']
        self.n_ent = params['n_ent']
        self.interval = params['interval']
        self.rela_embed = nn.Embedding(2*self.n_rel+2, 20)
        self.attention_1 = nn.Linear(60, 30, bias=False)
        self.attention_2 = nn.Linear(30, 1, bias=False)
        self.linear_classifier = nn.Linear(20, 1)
    
    def sample_data(self, dataset, t):
        data = []
        for i in range(t, t+self.interval):
            data.append(dataset[i])
        return np.concatenate(data, axis=0)

    # combine some time intervals
    def forward(self, X, dataset):
        batch_size = X.rel_idx.shape[0]
        query_rel = X.rel_idx
        cur_t = X.ts
        entity = X.src_idx
        cur_entity = np.array([[i, entity[i]] for i in range(batch_size)], dtype=np.int32)
        hidden_embedding = torch.zeros((batch_size, 20), device=self.device)

        # Get new index
        node_new_index = coo_matrix((np.arange(1, cur_entity.shape[0]+1), (cur_entity[:, 0], cur_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.longlong).toarray()
        
        for i in range(self.n_layer):
            # Get neighbors
            relation_t = []
            now_time = cur_t + self.interval * (i - self.n_layer)
            for b_idx in range(batch_size):
                sampled_data = self.sample_data(dataset, now_time[b_idx])
                n_rel_t = sampled_data.shape[0]
                # n_rel_t = dataset[now_time[b_idx]].shape[0]
                batch_b_data = np.concatenate([b_idx * np.ones((n_rel_t, 1), dtype=np.int32), sampled_data], axis=1)
                relation_t.append(batch_b_data)
            relation_all_t = np.concatenate(relation_t, axis=0, dtype=np.int32)

            # diagonal matrix group by batch index
            adj_batch = coo_matrix(
                    (
                        np.ones(relation_all_t.shape[0]), 
                        (np.arange(relation_all_t.shape[0]), relation_all_t[:,1]+relation_all_t[:,0]*self.n_ent)
                    ), shape=(relation_all_t.shape[0], batch_size*self.n_ent)
                )
            # current entity to one-hot vector, also grounp by batch index as a vector
            batch_data_sp_vec = coo_matrix((np.ones(cur_entity.shape[0]), (cur_entity[:,1]+cur_entity[:,0]*self.n_ent, np.zeros(cur_entity.shape[0]))),shape=(batch_size*self.n_ent, batch_size))
            select_one_hot = adj_batch @ batch_data_sp_vec
            select_idx = np.nonzero(select_one_hot)
            select_relation = relation_all_t[select_idx[0]]
            # embed_rel = hidden_embedding[node_new_index[select_relation[:,0]], :] + rel_embedding[select_relation[:,1], :]
            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed(torch.tensor(select_relation[:,2], device=self.device))


            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed(torch.tensor(select_relation[:,2], device=self.device)), self.rela_embed(torch.tensor(query_rel[select_relation[:, 0]], device=self.device))], dim=1)
            attention_score = torch.sigmoid(self.attention_2(F.relu(self.attention_1(attention_input))))

            # get unique (batch_idx, entity) as neighbours,
            # node_new_index: from entity to new index (how are embeddings listed)
            new_entity = np.unique(select_relation[:, [0, 3]], axis=0)
            node_new_index = coo_matrix((np.arange(1, new_entity.shape[0]+1), (new_entity[:, 0], new_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.longlong).toarray()


            row_col = np.array([node_new_index[select_relation[:,0], select_relation[:,3]]-1, np.arange(select_relation.shape[0])])

            sparse_matrix = torch.sparse_coo_tensor(row_col, attention_score.squeeze(), size=(new_entity.shape[0], select_relation.shape[0]), device=self.device)
            # sparse_matrix = torch.sparse_coo_tensor(row_col, torch.ones(select_relation.shape[0]), size=(new_entity.shape[0], select_relation.shape[0]))
            hidden_embedding = torch.sparse.mm(sparse_matrix, embed_rel)

            cur_entity = new_entity
            # print('ok')

        result = self.linear_classifier(hidden_embedding).reshape(-1)
        score_all = torch.zeros(batch_size, self.n_ent, device=self.device)
        score_all[cur_entity[:, 0], cur_entity[:, 1]] = result
        return score_all

class T_RED_GNN(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN, self).__init__()
        self.n_layer = params['n_layer']
        self.hidden_dim = params['hidden_dim']
        self.attn_dim = params['attn_dim']
        self.n_rel = params['n_rel']
        self.device = params['device']
        self.n_ent = params['n_ent']
        self.rela_embed = nn.Embedding(2*self.n_rel+2, 20)
        self.attention_1 = nn.Linear(60, 30, bias=False)
        self.attention_2 = nn.Linear(30, 1, bias=False)
        self.linear_classifier = nn.Linear(20, 1)

    # combine some time intervals
    def forward(self, X, dataset):
        entity, query_rel = X
        batch_size = len(entity)
        cur_entity = np.array([[i, entity[i]] for i in range(batch_size)], dtype=np.int32)
        hidden_embedding = torch.zeros((batch_size, 20), device=self.device)

        # Get new index
        node_new_index = coo_matrix((np.arange(1, cur_entity.shape[0]+1), (cur_entity[:, 0], cur_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.longlong).toarray()
        
        for i in range(3):
            # Get neighbors
            # diagonal matrix group by batch index
            random_idx = np.random.randint(dataset.shape[0], size=int(0.5 * dataset.shape[0]))
            dataset = dataset[random_idx, :]
            adj_batch = coo_matrix(
                    (
                        np.ones(dataset.shape[0]), 
                        (np.arange(dataset.shape[0]), dataset[:,0])
                    ), shape=(dataset.shape[0], self.n_ent)
                )
            # current entity to one-hot vector, also grounp by batch index as a vector
            batch_data_sp_vec = coo_matrix((np.ones(cur_entity.shape[0]), (cur_entity[:,1], cur_entity[:,0])),shape=(self.n_ent, batch_size))
            select_one_hot = adj_batch @ batch_data_sp_vec
            select_idx = np.nonzero(select_one_hot)
            select_relation = np.concatenate([select_idx[1].reshape(-1, 1), dataset[select_idx[0], :]], axis=1)
            # random_idx = np.random.randint(select_relation.shape[0], size=int(0.8 * select_relation.shape[0]))
            # select_relation = select_relation[random_idx, :]
            # embed_rel = hidden_embedding[node_new_index[select_relation[:,0]], :] + rel_embedding[select_relation[:,1], :]
            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed(torch.tensor(select_relation[:,2], device=self.device))


            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed(torch.tensor(select_relation[:,2], device=self.device)), self.rela_embed(torch.tensor(query_rel[select_relation[:, 0]], device=self.device))], dim=1)
            attention_score = torch.sigmoid(self.attention_2(F.relu(self.attention_1(attention_input))))

            # get unique (batch_idx, entity) as neighbours,
            # node_new_index: from entity to new index (how are embeddings listed)
            new_entity = np.unique(select_relation[:, [0, 3]], axis=0)
            node_new_index = coo_matrix((np.arange(1, new_entity.shape[0]+1), (new_entity[:, 0], new_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.longlong).toarray()


            row_col = np.array([node_new_index[select_relation[:,0], select_relation[:,3]]-1, np.arange(select_relation.shape[0])])

            sparse_matrix = torch.sparse_coo_tensor(row_col, attention_score.squeeze(), size=(new_entity.shape[0], select_relation.shape[0]), device=self.device)
            # sparse_matrix = torch.sparse_coo_tensor(row_col, torch.ones(select_relation.shape[0]), size=(new_entity.shape[0], select_relation.shape[0]))
            hidden_embedding = torch.sparse.mm(sparse_matrix, embed_rel)

            cur_entity = new_entity
            # print('ok')
        
        result = self.linear_classifier(hidden_embedding).reshape(-1)
        score_all = torch.zeros(batch_size, self.n_ent, device=self.device)
        score_all[cur_entity[:, 0], cur_entity[:, 1]] = result
        return score_all