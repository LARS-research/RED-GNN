import os
import sys
import time
from typing import List
import pdb
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from segment import segment_softmax_op_v2, segment_topk, segment_norm_l1_part, segment_norm_l1


class TimeEncode(torch.nn.Module):
    '''
    This class implemented the Bochner's time embedding
    expand_dim: int, dimension of temporal entity embeddings
    enitity_specific: bool, whether use entith specific freuency and phase.
    num_entities: number of entities.
    '''

    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        """
        :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
        :param entity_specific: if use entity specific time embedding
        :param num_entities: number of entities
        refer to Self-attention with Functional Time Representation Learning for more detail
        """
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific

        if entity_specific:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_entities, 1))
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_entities, 1))
        else:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        '''
        :param ts: [batch_size, seq_len]
        :param entities: which entities do we extract their time embeddings.
        :return: [batch_size, seq_len, time_dim]
        '''
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = torch.unsqueeze(ts, dim=2)
        # print("Forward in TimeEncode: ts is on ", ts.get_device())
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(
                dim=1)  # self.basis_freq[entities]:  [batch_size, time_dim]
            map_ts += self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [batch_size, 1, time_dim]
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic

class G3(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        """[summary]
        bilinear mapping along last dimension of x and y:
        output = MLP_1(x)^T A MLP_2(y), where A is two-dimenion matrix

        Arguments:
            left_dims {[type]} -- input dims of MLP_1
            right_dims {[type]} -- input dims of MLP_2
            output_dims {[type]} -- [description]
        """
        super(G3, self).__init__()
        self.dim_out = dim_out
        self.query_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.query_proj.weight, mean=0, std=np.sqrt(2.0 / (dim_in)))
        self.key_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.key_proj.weight, mean=0, std=np.sqrt(2.0 / (dim_in)))

    def forward(self, inputs):
        """[summary]
        Arguments:
            inputs: (left, right)
            left[i] -- tensor, bs x ... x left_dims
            right[i] -- tensor, bs x ... x right_dims
        """
        vi, vj = inputs
        left_x = torch.cat(vi, dim=-1)
        right_x = torch.cat(vj, dim=-1)
        # speed of batch-wise dot production: sum over element-wise product > matmul > bmm
        # refer to https://discuss.pytorch.org/t/dot-product-batch-wise/9746/12
        return torch.sum(self.query_proj(left_x) * self.key_proj(right_x), dim=-1) #/ np.sqrt(self.dim_out)

class AttentionFlow(nn.Module):
    def __init__(self, n_dims_in, n_dims_out, ratio_update=0, node_score_aggregation='sum',
                 device='cpu'):
        """[summary]

        Arguments:
            n_dims -- int, dimension of entity and relation embedding
            n_dims_sm -- int, smaller than n_dims to reduce the compuation consumption of calculating attention score
            ratio_update -- new node representation = ratio*self+(1-ratio)\sum{aggregation of neighbors' representation}
        """
        super(AttentionFlow, self).__init__()
        self.transition_fn = G3(4 * n_dims_in, 4 * n_dims_in)

        # dense layer between steps
        self.linear_between_steps = nn.Linear(n_dims_in, n_dims_out, bias=True)
        torch.nn.init.xavier_normal_(self.linear_between_steps.weight)
        self.act_between_steps = torch.nn.LeakyReLU()

        self.node_score_aggregation = node_score_aggregation
        self.ratio_update = ratio_update

        self.query_src_ts_emb = None
        self.query_rel_emb = None

        self.device = device

    def set_query_emb(self, query_src_ts_emb, query_rel_emb):
        self.query_src_ts_emb, self.query_rel_emb = query_src_ts_emb, query_rel_emb

    def _topk_att_score(self, edges, logits, k: int, tc=None):
        """

        :param edges: numpy array, (eg_idx, vi, ti, vj, tj, rel, node_idx_i, node_idx_j), dtype np.int32
        :param logits: tensor, same length as edges, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        pruned_edges, numpy.array, (eg_idx, vi, ts)
        pruned_logits, tensor, same length as pruned_edges
        origin_indices
        """
        if tc:
            t_start = time.time()
        res_edges = []
        res_logits = []
        res_indices = []
        for eg_idx in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg_idx
            orig_indices = np.arange(len(edges))[mask]
            masked_edges = edges[mask]
            masked_edges_logits = logits[mask]
            if masked_edges.shape[0] <= k:
                res_edges.append(masked_edges)
                res_logits.append(masked_edges_logits)
                res_indices.append(orig_indices)
            else:
                topk_edges_logits, indices = torch.topk(masked_edges_logits, k)
                res_indices.append(orig_indices[indices.cpu().numpy()])
                # pdb.set_trace()
                try:
                    res_edges.append(masked_edges[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_logits.append(topk_edges_logits)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_edges, axis=0), torch.cat(res_logits, dim=0), np.concatenate(res_indices, axis=0)

    def _cal_attention_score(self, edges, memorized_embedding, rel_emb):
        """
        calculating node attention from memorized embedding
        """
        hidden_vi_orig = memorized_embedding[edges[:, -2]]
        hidden_vj_orig = memorized_embedding[edges[:, -1]]

        return self.cal_attention_score(edges[:, 0], hidden_vi_orig, hidden_vj_orig, rel_emb)

    def cal_attention_score(self, query_idx, hidden_vi, hidden_vj, rel_emb):
        """
        calculate attention score between two nodes of edges
        wraped as a separate method so that it can be used for calculating attention between a node and it's full
        neighborhood, attention is used to select important nodes from the neighborhood
        :param query_idx: indicating in subgraph for which query the edge lies.
        """

        # [embedding]_repeat is a new tensor which index [embedding] so that it mathes hidden_vi and hidden_vj along dim 0
        # i.e. hidden_vi[i] and hidden_vj[i] is representation of node vi, vj that lie in subgraph corresponding to the query,
        # whose src, rel, time embedding is [embedding]_repeat[i]
        # [embedding] is one of query_src, query_rel, query_time
        query_src_ts_emb_repeat = torch.index_select(self.query_src_ts_emb, dim=0,
                                                     index=torch.from_numpy(query_idx).long().to(
                                                         self.device))
        query_rel_emb_repeat = torch.index_select(self.query_rel_emb, dim=0,
                                                  index=torch.from_numpy(query_idx).long().to(
                                                      self.device))

        transition_logits = self.transition_fn(
            ((hidden_vi, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat),
             (hidden_vj, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat)))

        return transition_logits

    def forward(self, visited_node_score, selected_edges_l=None, visited_node_representation=None, rel_emb_l=None,
                max_edges=10, analysis=False, tc=None):
        """calculate attention score

        Arguments:
            node_attention {tensor, num_edges} -- src_attention of selected_edges, node_attention[i] is the attention score
            of (selected_edge[i, 1], selected_edge[i, 2]) in eg_idx==selected_edge[i, 0]

        Keyword Arguments:
            selected_edges {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None})
            contain selfloop
            memorized_embedding torch.Tensor,
        return:
            pruned_edges, orig_indices
            updated_memorized_embedding:
            updated_node_score: Tensor, shape: n_new_node
            :param attended_nodes: 
        """
        updated_edge_attention = []  # for analysis

        transition_logits = self._cal_attention_score(selected_edges_l[-1], visited_node_representation, rel_emb_l[-1])

        # prune edges whose target node score is small
        # get source attention score
        src_score = visited_node_score[selected_edges_l[-1][:, -2]]
        transition_logits_softmax = segment_softmax_op_v2(transition_logits, selected_edges_l[-1][:, -2],
                                                          tc=tc)  # TB Check
        edge_attn_before_pruning = transition_logits_softmax  # for analysis
        target_score = transition_logits_softmax * src_score
        pruned_edges, pruned_target_score, orig_indices = self._topk_att_score(selected_edges_l[-1], target_score,
                                                                               max_edges)
        pruned_src_score = src_score[orig_indices]

        # transition_logits_pruned_softmax contains information of all selected_edges
        transition_logits_pruned_softmax = transition_logits_softmax[orig_indices]
        updated_edge_attention.append(transition_logits_pruned_softmax)

        num_nodes = len(visited_node_representation)
        if self.node_score_aggregation == 'max':
            max_dict = dict()
            for i in range(len(pruned_edges)):
                score_i = pruned_target_score[i].cpu().detach().numpy()
                if score_i > max_dict.get(pruned_edges[i, -1], (0, 0))[1]:
                    max_dict[pruned_edges[i, -1]] = (i, score_i)

            # biggest score from all edges (some edges may have the same subject)
            sparse_index = torch.LongTensor(
                np.stack([np.array(list(max_dict.keys())), np.array([_[0] for _ in max_dict.values()])])).to(
                self.device)
            trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, torch.ones(len(max_dict)).to(self.device),
                                                           torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            updated_node_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, pruned_target_score.unsqueeze(1)))
        elif self.node_score_aggregation in ['mean', 'sum']:
            sparse_index = torch.LongTensor(np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])).to(
                self.device)

            # node score aggregation
            if self.node_score_aggregation == 'mean':
                c = Counter(pruned_edges[:, -1])
                target_node_cnt = torch.tensor([c[_] for _ in pruned_edges[:, -1]]).to(self.device)
                transition_logits_pruned_softmax = torch.div(transition_logits_pruned_softmax, target_node_cnt)

            trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, transition_logits_pruned_softmax,
                                                           torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            # ATTENTION: updated_node_score[i] must be node score of node with node_idx==i
            updated_node_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, pruned_src_score.unsqueeze(1)))
        else:
            raise ValueError("node score aggregate can only be mean, sum or max")

        # only message passing and aggregation
        updated_visited_node_representation = self._update_node_representation_along_edges(pruned_edges,
                                                                                           visited_node_representation,
                                                                                           transition_logits_pruned_softmax,
                                                                                           linear_act=False)

        for selected_edges, rel_emb in zip(selected_edges_l[:-1][::-1], rel_emb_l[:-1][::-1]):
            transition_logits = self._cal_attention_score(selected_edges, updated_visited_node_representation,
                                                          rel_emb)
            transition_logits_softmax = segment_softmax_op_v2(transition_logits, selected_edges[:, -2], tc=tc)
            updated_edge_attention.append(transition_logits_softmax)
            updated_visited_node_representation = self._update_node_representation_along_edges(selected_edges,
                                                                                               updated_visited_node_representation,
                                                                                               transition_logits_softmax,
                                                                                               linear_act=False)

        #apply dense layer and activation on updated_memorized_embedding
        updated_visited_node_representation = self.bypass_forward(updated_visited_node_representation)

        if analysis:
            return updated_node_score, updated_visited_node_representation, pruned_edges, orig_indices, edge_attn_before_pruning, updated_edge_attention[
                                                                                                                                  ::-1]
        else:
            return updated_node_score, updated_visited_node_representation, pruned_edges, orig_indices

    def _update_node_representation_along_edges_old(self, edges, memorized_embedding, transition_logits):
        num_nodes = len(memorized_embedding)
        # update representation of nodes with neighbors
        # 1. message passing and aggregation
        sparse_index_rep = torch.from_numpy(edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sparse_value_rep = transition_logits
        trans_matrix_sparse_rep = torch.sparse.FloatTensor(sparse_index_rep.t(), sparse_value_rep,
                                                           torch.Size([num_nodes, num_nodes])).to(self.device)
        updated_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_rep, memorized_embedding)
        # 2. linear
        updated_memorized_embedding = self.act_between_steps(self.linear_between_steps(updated_memorized_embedding))
        # 3. pass representation of nodes without neighbors, i.e. not updated
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, -2])).unsqueeze(
            1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)
        trans_matrix_sparse_identical = torch.sparse.FloatTensor(sparse_index_identical.t(), sparse_value_identical,
                                                                 torch.Size([num_nodes, num_nodes])).to(self.device)
        identical_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_identical, memorized_embedding)
        updated_memorized_embedding = updated_memorized_embedding + identical_memorized_embedding
        return updated_memorized_embedding

    def _update_node_representation_along_edges(self, edges, node_representation, transition_logits, linear_act=True):
        """

        :param edges:
        :param memorized_embedding:
        :param transition_logits:
        :param linear_act: whether apply linear and activation layer after message aggregation
        :return:
        """
        num_nodes = len(node_representation)
        sparse_index_rep = torch.from_numpy(edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sparse_value_rep = (1 - self.ratio_update) * transition_logits
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, -2])).unsqueeze(
            1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)
        sparse_index_self = torch.from_numpy(np.unique(edges[:, -2])).unsqueeze(1).repeat(1, 2).to(self.device)
        sparse_value_self = self.ratio_update * torch.ones(len(sparse_index_self)).to(self.device)
        sparse_index = torch.cat([sparse_index_rep, sparse_index_identical, sparse_index_self], axis=0)
        sparse_value = torch.cat([sparse_value_rep, sparse_value_identical, sparse_value_self])
        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index.t(), sparse_value,
                                                       torch.Size([num_nodes, num_nodes])).to(self.device)
        updated_node_representation = torch.sparse.mm(trans_matrix_sparse, node_representation)
        if linear_act:
            updated_node_representation = self.act_between_steps(self.linear_between_steps(updated_node_representation))

        return updated_node_representation

    def bypass_forward(self, embedding):
        return self.act_between_steps(self.linear_between_steps(embedding))


class xERTE(torch.nn.Module):
    def __init__(self, ngh_finder, num_entity=None, num_rel=None, emb_dim: List[int] = None,
                 DP_num_edges=40, DP_steps=3,
                 emb_static_ratio=1, diac_embed=False,
                 node_score_aggregation='sum', ent_score_aggregation='sum', max_attended_edges=20, ratio_update=0,
                 device='cpu', analysis=False, use_time_embedding=True, **kwargs):
        """[summary]

        Arguments:
            ngh_finder {[type]} -- an instance of NeighborFinder, find neighbors of a node from temporal KG
            according to TGAN scheme

        Keyword Arguments:
            num_entity {[type]} -- [description] (default: {None})
            num_rel {[type]} -- [description] (default: {None})
            embed_dim {[type]} -- [dimension of ERTKG embedding] (default: {None})
            attn_mode {str} -- [currently only prod is supported] (default: {'prod'})
            use_time {str} -- [use time embedding] (default: {'time'})
            agg_method {str} -- [description] (default: {'attn'})
            tgan_num_layers {int} -- [description] (default: {2})
            tgan_n_head {int} -- [description] (default: {4})
            null_idx {int} -- [description] (default: {0})
            drop_out {float} -- [description] (default: {0.1})
            seq_len {[type]} -- [description] (default: {None})
            max_attended_nodes {int} -- [max number of nodes in attending-from horizon] (default: {20})
            ratio_update: new node representation = ratio*self+(1-ratio)\sum{aggregation of neighbors' representation}
            device {str} -- [description] (default: {'cpu'})
        """
        super(xERTE, self).__init__()
        assert len(emb_dim) == DP_steps + 1

        self.DP_num_edges = DP_num_edges
        self.DP_steps = DP_steps
        self.use_time_embedding = use_time_embedding
        self.ngh_finder = ngh_finder

        self.temporal_embed_dim = [int(emb_dim[_] * 2 / (1 + emb_static_ratio)) for _ in range(DP_steps)]
        self.static_embed_dim = [emb_dim[_] * 2 - self.temporal_embed_dim[_] for _ in range(DP_steps)]

        self.entity_raw_embed = torch.nn.Embedding(num_entity, self.static_embed_dim[0]).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = torch.nn.Embedding(num_rel + 1, emb_dim[0]).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = num_rel  # index of relation "selfloop"
        self.att_flow_list = nn.ModuleList([AttentionFlow(emb_dim[_], emb_dim[_ + 1],
                                                          node_score_aggregation=node_score_aggregation,
                                                          ratio_update=ratio_update, device=device)
                                            for _ in range(DP_steps)])
        if use_time_embedding:
            self.node_emb_proj = nn.Linear(2 * emb_dim[0], emb_dim[0])
        else:
            self.node_emb_proj = nn.Linear(emb_dim[0], emb_dim[0])

        nn.init.xavier_normal_(self.node_emb_proj.weight)
        self.max_attended_edges = max_attended_edges

        if use_time_embedding:
            self.time_encoder = TimeEncode(expand_dim=self.temporal_embed_dim[0], entity_specific=diac_embed,
                                           num_entities=num_entity, device=device)
        self.ent_spec_time_embed = diac_embed

        self.device = device
        self.analysis = analysis
        self.ent_score_aggregation = ent_score_aggregation

    def set_init(self, src_idx_l, rel_idx_l, cut_time_l):
        # save query information
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.cut_time_l = cut_time_l
        self.sampled_edges_l = []
        self.rel_emb_l = []
        # for input queries/nodes, node_idx == eg_idx
        self.node2index = {(i, src, ts): i for i, (src, rel, ts) in
                           enumerate(zip(src_idx_l, rel_idx_l, cut_time_l))}  # (eg_idx, ent, ts) -> node_idx
        self.num_existing_nodes = len(src_idx_l)

        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device)
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)
        if self.use_time_embedding:
            if self.ent_spec_time_embed:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).to(torch.float32).to(self.device),
                    entities=self.src_idx_l)
            else:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).to(torch.float32).to(self.device))
            query_ts_emb = torch.squeeze(query_ts_emb, 1)
            query_src_ts_emb = self.node_emb_proj(torch.cat([query_src_emb, query_ts_emb], axis=1))
        else:
            query_src_ts_emb = self.node_emb_proj(query_src_emb)

        # init query_src_ts_emb and query_rel_emb for each AttentionFlow layer
        for i, att_flow in enumerate(self.att_flow_list):
            if i > 0:
                query_src_ts_emb = self.att_flow_list[i - 1].bypass_forward(query_src_ts_emb)
                query_rel_emb = self.att_flow_list[i - 1].bypass_forward(query_rel_emb)
            att_flow.set_query_emb(query_src_ts_emb, query_rel_emb)

    def initialize(self):
        """get initial node (entity+time) embedding and initial node score

        Returns:
            attending_nodes, np.array -- n_attending_nodes x 3, (eg_idx, entity_id, ts)
            attending_node_attention, np,array -- n_attending_nodes, (1,)
            memorized_embedding, dict ((entity_id, ts): TGAN_embedding)
        """
        eg_idx_l = np.arange(len(self.src_idx_l), dtype=np.int32)
        att_score = np.ones_like(self.src_idx_l, dtype=np.float32) * (1 - 1e-8)

        attended_nodes = np.stack([eg_idx_l, self.src_idx_l, self.cut_time_l, np.arange(len(self.src_idx_l))], axis=1)
        visited_nodes_score = torch.from_numpy(att_score).to(self.device)
        visited_nodes = attended_nodes

        visited_node_representation = self.att_flow_list[0].query_src_ts_emb
        return attended_nodes, visited_nodes, visited_nodes_score, visited_node_representation

    def forward(self, sample):
        src_idx_l, rel_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.ts
        self.set_init(src_idx_l, rel_idx_l, cut_time_l)
        attended_nodes, visited_nodes, visited_node_score, visited_node_representation = self.initialize()
        for step in range(self.DP_steps):
            #            print("{}-th DP step".format(step))
            attended_nodes, visited_nodes, visited_node_score, visited_node_representation = \
                self._flow(attended_nodes, visited_nodes, visited_node_score, visited_node_representation, step)
            # pdb.set_trace()
            visited_node_score = segment_norm_l1(visited_node_score, visited_nodes[:, 0])
        entity_att_score, entities = self.get_entity_attn_score(visited_node_score[attended_nodes[:, -1]],
                                                                attended_nodes)

        return entity_att_score, entities

    def _flow(self, attended_nodes, visited_nodes, visited_node_score, visited_node_representation, step, tc=None):
        """[summary]
        Arguments:
            visited_nodes {numpy.array} -- num_nodes_visited x 4 (eg_idx, entity_id, ts, node_idx), dtype: numpy.int32, sort (eg_idx, ts, entity_id)
            all nodes visited during the expansion
            visited_node_score {Tensor} -- num_nodes_visited, dtype: torch.float32
            visited_node_representation {Tensor} -- num_nodes_visited x emb_dim_l[step]
            visited_node_score[node_idx] is the prediction score of node_idx
            visited_node_representation[node_idx] is the hidden representation of node_idx
        return:
            pruned_node {numpy.array} -- num_nodes_ x 4 (eg_idx, entity_id, ts, node_idx) sorted by (eg_idx, ts, entity_id)
            new_node_score {Tensor} -- new num_nodes_visited
            so that new_node_score[i] is the node prediction score of??
            updated_visited_node_representation: Tensor -- num_nodes_visited x emb_dim_l[step+1]
        """

        # Sampling Horizon
        # sampled_edges: (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj)
        # selfloop is added
        sampled_edges, new_sampled_nodes, new_attended_nodes = self._get_sampled_edges(attended_nodes,
                                                                                       num_neighbors=self.DP_num_edges,
                                                                                       step=step,
                                                                                       add_self_loop=True, tc=tc)
        if len(new_sampled_nodes):
            new_sampled_nodes_emb = self.get_node_emb(new_sampled_nodes[:, 1], new_sampled_nodes[:, 2],
                                                      eg_idx=new_sampled_nodes[:, 0])
            for i in range(step):
                new_sampled_nodes_emb = self.att_flow_list[i].bypass_forward(new_sampled_nodes_emb)
            visited_node_representation = torch.cat([visited_node_representation, new_sampled_nodes_emb], axis=0)
            visited_nodes = np.concatenate([visited_nodes, new_sampled_nodes], axis=0)

            assert len(visited_node_representation) == self.num_existing_nodes
            assert max(new_sampled_nodes[:, -1]) + 1 == self.num_existing_nodes
            assert max(sampled_edges[:, -1]) < self.num_existing_nodes

        self.sampled_edges_l.append(sampled_edges)

        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)
        for i in range(step):
            rel_emb = self.att_flow_list[i].bypass_forward(rel_emb)
        # update relation representation of edges sampled from previous steps
        for j in range(step):
            self.rel_emb_l[j] = self.att_flow_list[step - 1].bypass_forward(self.rel_emb_l[j])
        self.rel_emb_l.append(rel_emb)

        new_visited_node_score, updated_visited_node_representation, pruned_edges, orig_indices = \
            self.att_flow_list[step](visited_node_score,
                                     selected_edges_l=self.sampled_edges_l,
                                     visited_node_representation=visited_node_representation,
                                     rel_emb_l=self.rel_emb_l,
                                     max_edges=self.max_attended_edges, tc=tc)

        assert len(pruned_edges) == len(orig_indices)

        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]

        # get pruned nodes
        _, indices = np.unique(pruned_edges[:, [0, 4, 3]], return_index=True, axis=0)
        updated_attended_nodes = pruned_edges[:, [0, 3, 4, 7]][indices]

        return updated_attended_nodes, visited_nodes, new_visited_node_score, updated_visited_node_representation

    def loss(self, entity_att_score, entities, target_idx_l, batch_size, gradient_iters_per_update=1, loss_fn='BCE'):
        one_hot_label = torch.from_numpy(
            np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(self.device)
        try:
            assert gradient_iters_per_update > 0
            entity_att_score = entity_att_score*0.999+0.0009
            if loss_fn == 'BCE':
                if gradient_iters_per_update == 1:
                    loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
                else:
                    loss = torch.nn.BCELoss(reduction='sum')(entity_att_score, one_hot_label)
                    loss /= gradient_iters_per_update * batch_size
            else:
                # CE has problems
                if gradient_iters_per_update == 1:
                    loss = torch.nn.NLLLoss()(entity_att_score, one_hot_label)
                else:
                    loss = torch.nn.NLLLoss(reduction='sum')(entity_att_score, one_hot_label)
                    loss /= gradient_iters_per_update * batch_size
        except:
            print(entity_att_score)
            entity_att_score_np = entity_att_score.cpu().detach().numpy()
            print("all entity score smaller than 1:", all(entity_att_score_np < 1))
            print("all entity score greater than 0:", all(entity_att_score_np > 0))
            raise ValueError("Check if entity score in (0,1)")
        return loss

    def get_node_emb(self, src_idx_l, cut_time_l, eg_idx):

        hidden_node = self.get_ent_emb(src_idx_l, self.device)
        if self.use_time_embedding:
            cut_time_l = cut_time_l - self.cut_time_l[eg_idx]
            if self.ent_spec_time_embed:
                hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device),
                                                entities=src_idx_l)
            else:
                hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device))
            return self.node_emb_proj(torch.cat([hidden_node, torch.squeeze(hidden_time, 1)], axis=1))
        else:
            return self.node_emb_proj(hidden_node)

    def get_entity_attn_score(self, logits, nodes, tc=None):
        if tc:
            t_start = time.time()
        entity_attn_score, entities = self._aggregate_op_entity(logits, nodes, self.ent_score_aggregation)
        #        # normalize entity prediction score
        #        entity_attn_score = segment_norm_l1(entity_attn_score, entities[:, 0])
        if tc:
            tc['model']['entity_attn'] = time.time() - t_start
        return entity_attn_score, entities

    def _aggregate_op_entity(self, logits, nodes, aggr='sum'):
        """aggregate attention score of same entity, i.e. same (eg_idx, v)

        Arguments:
            logits {Tensor} -- attention score
            nodes {numpy.array} -- shape len(logits) x 3, (eg_idx, v, t), sorted by eg_idx, v, t
        return:
            entity_att_score {Tensor}: shape num_entity
            entities: numpy.array -- shape num_entity x 2, (eg_idx, v)
            att_score[i] if the attention score of entities[i]
        """
        device = logits.get_device()
        if device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        num_nodes = len(nodes)
        entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
        sparse_index = torch.LongTensor(np.stack([entities_idx, np.arange(num_nodes)]))
        sparse_value = torch.ones(num_nodes, dtype=torch.float)
        if aggr == 'mean':
            c = Counter([(node[0], node[1]) for node in nodes[:, :2]])
            target_node_cnt = torch.tensor([c[(_[0], _[1])] for _ in nodes[:, :2]])
            sparse_value = torch.div(sparse_value, target_node_cnt)

        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                       torch.Size([len(entities), num_nodes])).to(device)
        entity_att_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1)))

        return entity_att_score, entities

    def _get_sampled_edges(self, attended_nodes, num_neighbors: int = 20, step=None, add_self_loop=True, tc=None):
        """[summary]
        sample neighbors for attended_nodes from all events happen before attended_nodes
        with strategy specified by ngh_finder, selfloop is added
        attended nodes: nodes in the current subgraph
        Arguments:
            attended_nodes {numpy.array} shape: num_attended_nodes x 4 (eg_idx, vi, ti, node_idx), dtype int32
            -- [nodes (with time) in attended from horizon, for detail refer to ERTKG paper]

        Returns:
            sampled_edges: {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None}), sorted ascending by eg_idx, ti, vi, tj, vj, rel dtype int32
            new_sampled_nodes: {Tensor} shape: new_sampled_nodes
        """
        if tc:
            t_start = time.time()
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        node_idx_l = attended_nodes[:, 3]

        # src_ngh_x_batch: len(src_idx_l) x num_neighbors
        # concat(src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch) is ordered by (t, node, edge) ascending
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l,
            cut_time_l,
            num_neighbors=num_neighbors)

        if self.ngh_finder.sampling == -1:  # full neighborhood, select neighbors with largest attention score
            assert step is not None

            selected_src_ngh_node_batch = []
            selected_src_ngh_eidx_batch = []
            selected_src_ngh_t_batch = []
            with torch.no_grad():
                for i in range(len(src_ngh_eidx_batch)):
                    src_ngh_nodes = src_ngh_eidx_batch[i]
                    if sum(src_ngh_nodes != -1) > num_neighbors:

                        mask = (src_ngh_nodes != -1)
                        src_ngh_nodes = src_ngh_nodes[mask]
                        src_ngh_eidx = src_ngh_eidx_batch[i][mask]
                        src_ngh_t = src_ngh_t_batch[i][mask]
                        src_node_embed = self.get_node_emb(np.array([src_idx_l[i]] * len(src_ngh_nodes)),
                                                           np.array([cut_time_l[i]] * len(src_ngh_nodes)),
                                                           np.array([attended_nodes[i, 0] * len(src_ngh_nodes)]))
                        ngh_node_embed = self.get_node_emb(src_ngh_nodes, src_ngh_t,
                                                           np.array([attended_nodes[i, 0] * len(src_ngh_nodes)]))
                        rel_emb = self.get_rel_emb(src_ngh_eidx, self.device)

                        att_scores = self.att_flow_list[step].cal_attention_score(
                            np.ones(len(src_ngh_nodes)) * attended_nodes[i, 0], src_node_embed, ngh_node_embed, rel_emb)
                        _, indices = torch.topk(att_scores, num_neighbors)
                        indices = indices.cpu().numpy()
                        indices_sorted_by_timestamp = sorted(indices, key=lambda x: (
                            src_ngh_t[x], src_ngh_nodes[x], src_ngh_eidx[x]))
                        selected_src_ngh_node_batch.append(src_ngh_nodes[indices_sorted_by_timestamp])
                        selected_src_ngh_eidx_batch.append(src_ngh_eidx[indices_sorted_by_timestamp])
                        selected_src_ngh_t_batch.append(src_ngh_t[indices_sorted_by_timestamp])
                    else:
                        selected_src_ngh_node_batch.append(src_ngh_nodes[-num_neighbors:])
                        selected_src_ngh_eidx_batch.append(src_ngh_eidx_batch[i][-num_neighbors:])
                        selected_src_ngh_t_batch.append(src_ngh_t_batch[i][-num_neighbors:])
                src_ngh_node_batch = np.stack(selected_src_ngh_node_batch)
                src_ngh_eidx_batch = np.stack(selected_src_ngh_eidx_batch)
                src_ngh_t_batch = np.stack(selected_src_ngh_t_batch)

        # add selfloop
        if add_self_loop:
            src_ngh_node_batch = np.concatenate([src_ngh_node_batch, src_idx_l[:, np.newaxis]], axis=1)
            src_ngh_eidx_batch = np.concatenate(
                [src_ngh_eidx_batch, np.array([[self.selfloop] for _ in range(len(attended_nodes))], dtype=np.int32)],
                axis=1)
            src_ngh_t_batch = np.concatenate([src_ngh_t_batch, cut_time_l[:, np.newaxis]], axis=1)
        # removed padded neighbors, with node idx == rel idx == -1
        src_ngh_node_batch_flatten = src_ngh_node_batch.flatten()
        src_ngh_eidx_batch_flatten = src_ngh_eidx_batch.flatten()
        src_ngh_t_batch_faltten = src_ngh_t_batch.flatten()
        eg_idx = np.repeat(attended_nodes[:, 0], num_neighbors + int(add_self_loop))
        mask = src_ngh_node_batch_flatten != -1

        # (eg_idx, src_ent, src_ts, tar_ent, tar_ts, rel, src_node_idx)
        sampled_edges = np.stack([eg_idx,
                                  np.repeat(src_idx_l, num_neighbors + int(add_self_loop)),
                                  np.repeat(cut_time_l, num_neighbors + int(add_self_loop)), \
                                  src_ngh_node_batch_flatten, src_ngh_t_batch_faltten, \
                                  src_ngh_eidx_batch_flatten,
                                  np.repeat(node_idx_l, num_neighbors + int(add_self_loop))], axis=1)[mask]

        # index new selected nodes
        target_nodes_index = []
        new_sampled_nodes = []
        for eg, tar_node, tar_ts in sampled_edges[:, [0, 3, 4]]:
            if (eg, tar_node, tar_ts) in self.node2index.keys():
                target_nodes_index.append(self.node2index[(eg, tar_node, tar_ts)])
            else:
                self.node2index[(eg, tar_node, tar_ts)] = self.num_existing_nodes
                target_nodes_index.append(self.num_existing_nodes)
                new_sampled_nodes.append([eg, tar_node, tar_ts, self.num_existing_nodes])
                self.num_existing_nodes += 1

        sampled_edges = np.concatenate([sampled_edges, np.array(target_nodes_index)[:, np.newaxis]], axis=1)
        # new_sampled_nodes sorted by node_idx so that visited_node_representation[i] is the node representation of node i
        new_sampled_nodes = sorted(new_sampled_nodes, key=lambda x: x[-1])
        new_sampled_nodes = np.array(new_sampled_nodes)

        # new_attended_nodes sorted by (eg_idx, t, ent)
        _, new_attended_nodes_idx = np.unique(sampled_edges[:, [0, 4, 3]], return_index=True, axis=0)
        new_attended_nodes = sampled_edges[:, [0, 3, 4]][new_attended_nodes_idx]

        if tc:
            tc['graph']['sample'] += time.time() - t_start
        return sampled_edges, new_sampled_nodes, new_attended_nodes

    def _topk_att_score(self, attending_nodes, attending_node_attention, k: int, tc=None):
        """

        :param attending_nodes: numpy array, N_visited_nodes x 4 (eg_idx, vi, ts, node_idx), dtype np.int32
        :param attending_node_attention: tensor, N_all_visited_nodes, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        attended_nodes, numpy.array, (eg_idx, vi, ts)
        attended_node_attention, tensor, attention_score, same length as attended_nodes
        attended_node_emb, tensor, same length as attended_nodes
        """
        if tc:
            t_start = time.time()
        res_nodes = []
        res_att = []
        attending_node_attention = attending_node_attention[
            torch.from_numpy(attending_nodes[:, 3]).to(torch.int64).to(self.device)]
        for eg_idx in sorted(set(attending_nodes[:, 0])):
            mask = attending_nodes[:, 0] == eg_idx
            masked_nodes = attending_nodes[mask]
            masked_node_attention = attending_node_attention[mask]
            if masked_nodes.shape[0] <= k:
                res_nodes.append(masked_nodes)
                res_att.append(masked_node_attention)
            else:
                topk_node_attention, indices = torch.topk(masked_node_attention, k)
                try:
                    res_nodes.append(masked_nodes[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_att.append(topk_node_attention)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_nodes, axis=0), torch.cat(res_att, dim=0)

    def get_ent_emb(self, ent_idx_l, device):
        """
        help function to get node embedding
        self.entity_raw_embed[0] is the embedding for dummy node, i.e. node non-existing

        Arguments:
            node_idx_l {np.array} -- indices of nodes
        """
        embed_device = next(self.entity_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.entity_raw_embed(torch.from_numpy(ent_idx_l).long().to(embed_device)).to(device)

    def get_rel_emb(self, rel_idx_l, device):
        """
        help function to get relation embedding
        self.edge_raw_embed[0] is the embedding for dummy relation, i.e. relation non-existing
        Arguments:
            rel_idx_l {[type]} -- [description]
        """
        embed_device = next(self.relation_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.relation_raw_embed(torch.from_numpy(rel_idx_l).long().to(embed_device)).to(device)
