import torch
from torch import embedding, nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, block_diag, coo_matrix
import numpy as np
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
import pickle
import networkx as nx
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyvis.network import Network
import math

# class Time_Embed(nn.Module):
#     def __init__(self, time_embed):
#         super(Time_Embed, self).__init__()
#         self.time_embed = time_embed
#         # self.time_linear = nn.Linear(10, self.time_embed)
#         self.time_linear_list = nn.ModuleList([nn.Linear(1, self.time_embed) for i in range(10)])
    
#     def forward(self, ts):
#         # ts = ts.repeat(1, 10)
#         # return torch.cos(self.time_linear(ts))
#         res = 0
#         for i in range(10):
#             res = res + torch.cos(self.time_linear_list[i](ts))
#         return res

class Time_Embed_1(nn.Module):
    def __init__(self, time_embed):
        super(Time_Embed_1, self).__init__()
        self.time_dim = time_embed
        self.basis_freq = torch.nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
    
    def forward(self, ts):
        ts = ts.view(-1, 1)
        return torch.cos(ts * self.basis_freq.view(1, -1) + self.phase.view(1, -1))

class Time_Embed_Entity(nn.Module):
    def __init__(self, time_embed, num_entity):
        super(Time_Embed_Entity, self).__init__()
        self.time_dim = time_embed
        self.basis_freq = torch.nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().repeat(num_entity, 1))  # shape: num_entities * time_dim
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().repeat(num_entity, 1))
    
    def forward(self, ts, entity):
        ts = ts.view(-1, 1)
        return torch.cos(ts * self.basis_freq[entity, :] + self.phase[entity, :])

class T_RED_GNN(nn.Module):
    def __init__(self, params):
        super(T_RED_GNN, self).__init__()
        self.n_rel = params.contents.num_relations+1
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

        self.time_embed = nn.Embedding(182, self.hidden_dim)
        # self.time_embed_fn = Time_Embed(self.hidden_dim)
        # self.time_embed = Time_Embed_Entity(self.hidden_dim, self.n_ent)
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
        self.init_params()
        self.attention_vis = []
        with open('id2entity.pickle', 'rb') as handle:
            self.id2entity = pickle.load(handle)
        
        with open('id2relation.pickle', 'rb') as handle:
            self.id2relation = pickle.load(handle)

    def init_params(self):
        for i in range(self.n_layer):
            nn.init.xavier_normal_(self.rela_embed_layer[i].weight)
        nn.init.xavier_normal_(self.time_embed.weight)
    
    def get_data(self, id, type):
        if type == 'entity':
            return self.id2entity[id]
        elif type == 'relation':
            return self.id2relation[id]
        elif type == 'time':
            return (datetime(2014, 1, 1) + timedelta(days=int(id)//24)).strftime('%Y-%m-%d')


    # combine some time intervals
    def forward(self, X):
        batch_size = X.rel_idx.shape[0]
        query_rel = torch.from_numpy(X.rel_idx).to(self.device)
        cur_t = torch.from_numpy(X.ts // 24).to(self.device)
        entity = torch.from_numpy(X.src_idx).to(self.device)
        cur_entity = torch.column_stack([torch.arange(batch_size, device=self.device), entity])
        hidden_embedding = torch.zeros((batch_size, self.hidden_dim), device=self.device)

        attention_dict = {
            id:[] for id in range(0, 365)
        }
        attention_dict['query_rel_time'] = X.ts[0] // 24

        rdigraphh = nx.DiGraph()
        source_ent, target_ent, rel_name = f'{self.get_data(X.src_idx[0], "entity")}_0', f'{self.get_data(X.target_idx[0], "entity")}_{self.n_layer}', f'{self.get_data(X.rel_idx[0], "relation")}_{self.get_data(X.ts[0], "time")}'

        # Get new index
        node_new_index = torch.sparse_coo_tensor(
            indices=cur_entity.T,
            values=torch.arange(1, cur_entity.shape[0]+1, device=self.device),
            size=(batch_size, self.n_ent),
            device=self.device,
            requires_grad=False
        ).to_dense()

        # prepare dataset
        # add selfloop
        relation_t = []
        for b_idx in range(batch_size):
            begin_time = (cur_t[b_idx] - 120).item()
            if begin_time < 0:
                begin_time = 0
            data_batch = self.dataset[self.time_offset_list[begin_time]:self.time_offset_list[cur_t[b_idx]]]
            selfloop_rel = np.column_stack([np.arange(self.n_ent), np.ones(self.n_ent, dtype=np.int64)*self.n_rel_true, np.arange(self.n_ent), begin_time * 24 * np.ones(self.n_ent, dtype=np.int64)])
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
        
        rdigraphh.add_node(source_ent, layer=0)

        for i in range(self.n_layer):
            batch_data_sp_vec = coo_matrix((np.ones(cur_entity.shape[0]), ((cur_entity[:,1]+cur_entity[:,0]*self.n_ent).tolist(), np.zeros(cur_entity.shape[0]))),shape=(batch_size*self.n_ent, 1))
            select_one_hot = adj_batch @ batch_data_sp_vec
            select_idx = np.nonzero(select_one_hot)
            select_relation = torch.from_numpy(relation_all_t[select_idx[0]]).to(self.device)

            # get embedding
            relative_time_index = cur_t[select_relation[:, 0]] - torch.div(select_relation[:, 4], 24, rounding_mode='floor')
            # relative_time_index = torch.div(select_relation[:, 4], 24, rounding_mode='floor').type(torch.float32)

            # time_embedding = self.time_embed(relative_time_index.type(torch.float32).view(-1, 1), entity[select_relation[:, 0]].long())
            # time_embedding = self.time_embed(relative_time_index.type(torch.float32).view(-1, 1))
            time_embedding = self.time_embed(relative_time_index.long())

            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed_layer[i](select_relation[:,2]) + time_embedding

            transformed_embed = self.past_linear(embed_rel)

            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed_layer[i](select_relation[:,2]), self.rela_embed_layer[i](query_rel[select_relation[:, 0]])], dim=1)
            attention_score = torch.sigmoid(self.attention_2_layer[i](F.relu(self.attention_1_layer[i](attention_input))))



            all_target_entity = set(select_relation[:, 3].cpu().numpy().tolist())
            for j in all_target_entity:
                rdigraphh.add_node(f'{self.get_data(j, "entity")}_{i+1}', layer=i+1)

            for j in tqdm(range(select_relation.shape[0])):
                if attention_score[j, 0].item() > torch.topk(attention_score[:, 0], math.ceil(0.05*attention_score.size(0)))[0][-1].item():
                    rdigraphh.add_edge(f'{self.get_data(select_relation[j, 1].item(), "entity")}_{i}', f'{self.get_data(select_relation[j, 3].item(), "entity")}_{i+1}', label=f'{self.get_data(select_relation[j, 2].item(), "relation")}_{self.get_data(select_relation[j, 4].item(), "time")}', weight=5*attention_score[j, 0].item(), title=f'{attention_score[j, 0].item()}')

            attention_dict[f'{i+1}layer']['attention_score'] = attention_score.cpu().detach().numpy().tolist()
            attention_dict[f'{i+1}layer']['rel'] = select_relation[:,2].cpu().numpy().tolist()


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
        # self.attention_vis.append(attention_dict)
        # with open('attention_vis.pickle', 'wb') as handle:
        #     pickle.dump(self.attention_vis, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # rdigraphh.add_edge(source_ent, target_ent, label=rel_name, weight=10, smooth=True)
        paths_between_generator = nx.all_simple_paths(rdigraphh, source=source_ent, target=target_ent)
        nodes_between_set = set()
        for path in paths_between_generator:
            for node in path:
                nodes_between_set.add(node)
        SG = rdigraphh.subgraph(list(nodes_between_set))

        # rdigraphh.add_edge(source_ent, target_ent, label=rel_name, weight=2, smooth=True)

        # c_score = nx.algorithms.betweenness_centrality_subset(rdigraphh, sources=(source_ent,), targets=(target_ent,))
        # nodes_between = [x for x in c_score if c_score[x]!=0.0]
        # nodes_between.extend((source_ent, target_ent))  #add on the ends
        # SG = rdigraphh.subgraph(nodes_between)

        # fig = plt.figure(1, figsize=(200, 80), dpi=60)
        # nx.draw_networkx(SG, pos=nx.multipartite_layout(SG, subset_key='layer'), with_labels=True, font_weight='bold', node_color='r', font_size=20, width=0.5, node_size=100)
        # nx.draw_networkx_edge_labels(SG, pos=nx.multipartite_layout(SG, subset_key='layer'), edge_labels=nx.get_edge_attributes(SG, 'label'), font_size=20)
        # plt.axis('off')
        # plt.savefig('test.pdf')
        # plt.clf()

        pos=nx.multipartite_layout(SG, subset_key='layer', scale=500)

        nt = Network()
        nt.from_nx(SG)
        # nt.add_edge(source_ent, target_ent, label=rel_name, weight=2, smooth=True)

        nt.show_buttons()
        for node in nt.get_nodes():
            nt.get_node(node)['x']=pos[node][0]*10
            nt.get_node(node)['y']=-pos[node][1]*2 #the minus is needed here to respect networkx y-axis convention 
            nt.get_node(node)['physics']=False
            nt.get_node(node)['label']=str(node) #set the node label as a string so that it can be displayed

        nt.add_node(1, shape='box', value=200,
                    # x=[21.4, 54.2, 11.2],
                    # y=[100.2, 23.54, 32.1],
                    label=f'{source_ent}, {rel_name}, {target_ent}',)
                    # color=['#00ff1e', '#162347', '#dd4b39'])
        

        # nt.toggle_physics(False)
        nt.show_buttons(filter_=['physics'])
        nt.save_graph('test.html')

        # nx.draw_networkx(rdigraphh, with_labels=True, font_weight='bold', node_color='r', font_size=8, width=0.5, node_size=100)
        # plt.savefig('full.pdf')
        # plt.clf()


        return score_all, (result, cur_entity.cpu().numpy())

    def unique(self, A):
        return torch.sparse_coo_tensor(
            indices=A.T,
            values=torch.ones(A.shape[0]),
            device=self.device,
            requires_grad=False
        ).coalesce().indices().T

class RED_GNN(nn.Module):
    def __init__(self, params):
        super(RED_GNN, self).__init__()
        self.n_rel = params.contents.num_relations+1
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
        # self.now_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.future_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # self.time_embed = nn.Embedding(182, self.hidden_dim)
        # self.time_embed_fn = Time_Embed(self.hidden_dim)
        # self.time_embed = Time_Embed_Entity(self.hidden_dim, self.n_ent)
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

        attention_dict = {
            'query_rel': query_rel.cpu().numpy().tolist(), 
            '1layer': {},
            '2layer': {},
            '3layer': {},
            '4layer': {},
        }

        # Get new index
        node_new_index = torch.sparse_coo_tensor(
            indices=cur_entity.T,
            values=torch.arange(1, cur_entity.shape[0]+1, device=self.device),
            size=(batch_size, self.n_ent),
            device=self.device,
            requires_grad=False
        ).to_dense()

        # prepare dataset
        # add selfloop
        relation_t = []
        for b_idx in range(batch_size):
            begin_time = (cur_t[b_idx] - 120).item()
            if begin_time < 0:
                begin_time = 0
            data_batch = self.dataset[self.time_offset_list[begin_time]:self.time_offset_list[cur_t[b_idx]]]
            selfloop_rel = np.column_stack([np.arange(self.n_ent), np.ones(self.n_ent, dtype=np.int64)*self.n_rel_true, np.arange(self.n_ent), begin_time * 24 * np.ones(self.n_ent, dtype=np.int64)])
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
            # relative_time_index = torch.div(select_relation[:, 4], 24, rounding_mode='floor').type(torch.float32)

            # time_embedding = self.time_embed(relative_time_index.type(torch.float32).view(-1, 1), entity[select_relation[:, 0]].long())
            # time_embedding = self.time_embed(relative_time_index.type(torch.float32).view(-1, 1))
            # time_embedding = self.time_embed(relative_time_index.long())

            embed_rel = hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :] + self.rela_embed_layer[i](select_relation[:,2])

            transformed_embed = self.past_linear(embed_rel)

            attention_input = torch.cat([hidden_embedding[node_new_index[select_relation[:,0], select_relation[:,1]]-1, :], self.rela_embed_layer[i](select_relation[:,2]), self.rela_embed_layer[i](query_rel[select_relation[:, 0]])], dim=1)
            attention_score = torch.sigmoid(self.attention_2_layer[i](F.relu(self.attention_1_layer[i](attention_input))))

            attention_dict[f'{i+1}layer']['attention_score'] = attention_score.cpu().numpy().tolist()
            attention_dict[f'{i+1}layer']['rel'] = select_relation[:,2].numpy().tolist()
            
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
