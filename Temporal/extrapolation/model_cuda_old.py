import torch
from torch import embedding, nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, block_diag, coo_matrix
import numpy as np
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

class Time_Embed(nn.Module):
    def __init__(self, time_embed):
        super(Time_Embed, self).__init__()
        self.time_embed = time_embed
        self.time_linear = nn.Linear(1, self.time_embed)
    
    def forward(self, ts):
        return torch.cos(self.time_linear(ts))

class Time_Embed_v2(nn.Module):
    def __init__(self, time_embed_size):
        super(Time_Embed_v2, self).__init__()
        self.time_embed_size = time_embed_size
        self.week_embed = nn.Embedding(7, self.time_embed_size)
        self.month_embed = nn.Embedding(30, self.time_embed_size)
        self.season_embed = nn.Embedding(120, self.time_embed_size)
    
    def forward(self, ts):
        week_idx = ts % 7
        month_idx = ts % 30
        season_idx = ts % 120
        return self.week_embed(week_idx) + self.month_embed(month_idx) + self.season_embed(season_idx)


class T_RED_GNN(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN, self).__init__()
        self.n_rel = params.contents.num_relations + 1
        self.n_rel_true = params.contents.num_relations
        self.device = params.device
        self.n_ent = params.contents.num_entities
        self.n_time = len(params.contents.timestamps)
        self.dataset = params.contents.data
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

        # self.time_embed = nn.Embedding(self.n_time, self.hidden_dim)
        self.time_embed = Time_Embed_v2(self.hidden_dim)
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
        self.time_offset_list = params.time_offset_list

    # combine some time intervals
    def forward(self, X):
        batch_size = X.rel_idx.shape[0]
        query_rel = torch.from_numpy(X.rel_idx).to(self.device)
        cur_t = torch.from_numpy(X.ts // 24).to(self.device)
        entity = torch.from_numpy(X.src_idx).to(self.device)
        cur_entity = torch.column_stack([torch.arange(batch_size, device=self.device), entity])
        hidden_embedding = torch.zeros((batch_size, self.hidden_dim), device=self.device)

        # Get new index
        node_new_index = torch.sparse_coo_tensor(
            indices=cur_entity.T,
            values=torch.arange(1, cur_entity.shape[0]+1, device=self.device),
            size=(batch_size, self.n_ent),
            device=self.device,
            requires_grad=False
        ).to_dense()
        
        # prepare dataset
        relation_t = []
        # add selfloop
        selfloop_rel = np.column_stack([np.arange(self.n_ent), np.ones(self.n_ent, dtype=np.int64)*self.n_rel_true, np.arange(self.n_ent), np.zeros(self.n_ent, dtype=np.int64)])
        for b_idx in range(batch_size):
            data_batch = self.dataset[:self.time_offset_list[cur_t[b_idx]]]
            data_batch = np.concatenate([selfloop_rel, data_batch], axis=0)
            n_rel_t = data_batch.shape[0]
            data_batch = np.concatenate([b_idx * np.ones((n_rel_t, 1), dtype=np.int64), data_batch], axis=1)
            relation_t.append(data_batch)
        relation_all_t = np.concatenate(relation_t, axis=0, dtype=np.int64)

        adj_batch = coo_matrix(
                (
                    np.ones(relation_all_t.shape[0]), 
                    (np.arange(relation_all_t.shape[0]), relation_all_t[:,1]+relation_all_t[:,0]*self.n_ent)
                ), shape=(relation_all_t.shape[0], batch_size*self.n_ent)
            )

        for i in range(self.n_layer):
            batch_data_sp_vec = coo_matrix((np.ones(cur_entity.shape[0]), ((cur_entity[:,1]+cur_entity[:,0]*self.n_ent).tolist(), np.zeros(cur_entity.shape[0]))),shape=(batch_size*self.n_ent, 1))
            select_one_hot = adj_batch @ batch_data_sp_vec
            select_idx = np.nonzero(select_one_hot)
            select_relation = torch.from_numpy(relation_all_t[select_idx[0]]).to(self.device)

            # get embedding
            # relative_time_index = cur_t[select_relation[:, 0]] - torch.div(select_relation[:, 4], 24, rounding_mode='floor')
            # relative_time_index = cur_t[select_relation[:, 0]].type(torch.float32)
            relative_time_index = torch.div(select_relation[:, 4], 24, rounding_mode='floor')

            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed_layer[i](select_relation[:,2]) + self.time_embed(relative_time_index)

            transformed_embed = self.past_linear(embed_rel)

            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed_layer[i](select_relation[:,2]), self.rela_embed_layer[i](query_rel[select_relation[:, 0]])], dim=1)
            attention_score = torch.sigmoid(self.attention_2_layer[i](F.relu(self.attention_1_layer[i](attention_input))))

            msg_pass = attention_score * transformed_embed

            new_entity = self.unique(select_relation[:, [0, 3]])

            node_new_index = torch.sparse_coo_tensor(
                indices=new_entity.T,
                values=torch.arange(1, new_entity.shape[0]+1, device=self.device),
                size=(batch_size, self.n_ent),
                device=self.device,
                requires_grad=False
            ).to_dense()

            new_index = node_new_index[select_relation[:,0], select_relation[:,3]]-1

            hidden_embedding = scatter(msg_pass, new_index, dim=0, reduce='sum')
            hidden_embedding = self.act(hidden_embedding)

            cur_entity = new_entity
        
        result = self.linear_classifier(hidden_embedding).reshape(-1)
        score_all = torch.zeros(batch_size, self.n_ent, device=self.device)
        score_all[cur_entity[:, 0], cur_entity[:, 1]] = result
        result = scatter_softmax(result, cur_entity[:,0])
        return score_all, (result, cur_entity.cpu().numpy())
    
    def unique(self, A):
        return torch.sparse_coo_tensor(
            indices=A.T,
            values=torch.ones(A.shape[0]),
            device=self.device,
            requires_grad=False
        ).coalesce().indices().T
