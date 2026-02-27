from enum import unique
from platform import node
from numpy import dtype, einsum
import torch
from torch import embedding, nn, rand
import torch.nn.functional as F
from bisect import bisect_left
from scipy.sparse import csr_matrix, block_diag, coo_matrix
import numpy as np
from torch_scatter import scatter

class T_RED_GNN(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN, self).__init__()
        self.n_rel = len(params.relation_vocab)
        self.device = params.device
        self.n_ent = len(params.entity_vocab)
        self.dataset = params.graph
        self.rela_embed = nn.Embedding(self.n_rel+1, 20)
        self.attention_1 = nn.Linear(60, 30, bias=False)
        self.attention_2 = nn.Linear(30, 1, bias=False)
        self.linear_classifier = nn.Linear(20, 1)

        self.past_linear = nn.Linear(20, 20, bias=False)
        self.now_linear = nn.Linear(20, 20, bias=False)
        self.future_linear = nn.Linear(20, 20, bias=False)

        self.time_embed = nn.Embedding(365, 20)

        # cover rate
        # self.sp2o = {}
        # for spot in self.dataset:
        #     sp = tuple(spot[:2])
        #     if sp not in self.sp2o.keys():
        #         self.sp2o[sp] = [spot[2]]
        #     else:
        #         self.sp2o[sp].append(spot[2])

    # combine some time intervals
    def forward(self, batch):
        batch_size = batch['head'].shape[0]
        query_rel = batch['relation'].cpu().numpy()
        entity = batch['head'].cpu().numpy()
        query_time = batch['time'].cpu().numpy()
        dataset = np.delete(self.dataset, batch['example_idx'], axis=0)
        cur_entity = np.array([[i, entity[i]] for i in range(batch_size)], dtype=np.int32)
        hidden_embedding = torch.zeros((batch_size, 20), device=self.device)
        adj_batch = coo_matrix(
                (
                    np.ones(dataset.shape[0]), 
                    (np.arange(dataset.shape[0]), dataset[:,0])
                ), shape=(dataset.shape[0], self.n_ent)
            )
        # Get new index
        node_new_index = coo_matrix((np.arange(1, cur_entity.shape[0]+1), (cur_entity[:, 0], cur_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.int64).toarray()
        
        for i in range(3):
            # Get neighbors

            # current entity to one-hot vector, also grounp by batch index as a vector
            batch_data_sp_vec = coo_matrix((np.ones(cur_entity.shape[0]), (cur_entity[:,1], cur_entity[:,0])),shape=(self.n_ent, batch_size))
            select_one_hot = adj_batch @ batch_data_sp_vec
            select_idx = np.nonzero(select_one_hot)
            select_relation = np.concatenate([select_idx[1].reshape(-1, 1), dataset[select_idx[0], :]], axis=1)

            # random_select_relation = np.random.randint(select_relation.shape[0], size=int(0.5 * select_relation.shape[0]))
            # select_relation = select_relation[random_select_relation, :]
            relative_time_index = select_relation[:, 4] - query_time[select_relation[:, 0]]

            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed(torch.from_numpy(select_relation[:,2]).to(self.device)) + self.time_embed(torch.from_numpy(np.abs(relative_time_index)).to(self.device))

            transformed_embed = torch.zeros_like(embed_rel, device=self.device)
            transformed_embed[relative_time_index > 0] = self.future_linear(embed_rel[relative_time_index > 0])
            transformed_embed[relative_time_index == 0] = self.now_linear(embed_rel[relative_time_index == 0])
            transformed_embed[relative_time_index < 0] = self.past_linear(embed_rel[relative_time_index < 0])

            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed(torch.tensor(select_relation[:,2], device=self.device)), self.rela_embed(torch.tensor(query_rel[select_relation[:, 0]], device=self.device))], dim=1)
            attention_score = torch.sigmoid(self.attention_2(F.relu(self.attention_1(attention_input))))
            msg_pass = attention_score * transformed_embed

            # get unique (batch_idx, entity) as neighbours,
            # node_new_index: from entity to new index (how are embeddings listed)
            # new_entity = np.unique(select_relation[:, [0, 3]], axis=0)
            new_entity = self.unique(select_relation[:, [0, 3]])
            node_new_index = coo_matrix((np.arange(1, new_entity.shape[0]+1), (new_entity[:, 0], new_entity[:, 1])), shape=(batch_size, self.n_ent), dtype=np.int64).toarray()

            
            new_index = torch.from_numpy(node_new_index[select_relation[:,0], select_relation[:,3]]-1).to(self.device)
            hidden_embedding = scatter(msg_pass, new_index, dim=0, reduce='sum')
            hidden_embedding = F.leaky_relu(hidden_embedding)

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
        Sp = coo_matrix((np.ones(A.shape[0]), (A[:,0], A[:, 1]))).todense()
        coordinate = np.nonzero(Sp)
        return np.vstack(coordinate).T

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
