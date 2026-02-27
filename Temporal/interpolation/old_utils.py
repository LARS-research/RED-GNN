import os
import sys
import json
import subprocess
from collections import defaultdict
import numpy as np
import networkx as nx
import torch
from model import xERTE

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)
DataDir = os.path.join(os.path.dirname(__file__), 'data')

class Data:
    def __init__(self, dataset=None, add_reverse_relation=False):
        """
        :param dataset:
        :param add_reverse_relation: if True, add reversed relation
        """
        # load data
        self.id2entity = self._id2entity(dataset=dataset)
        self.id2relation = self._id2relation(dataset=dataset)
        num_relations = len(self.id2relation)  # number of original relations, i.e., no reversed relation
        reversed_id2relation = {}
        if add_reverse_relation:
            for ind, rel in self.id2relation.items():
                reversed_id2relation[ind + num_relations] = 'Reversed ' + rel
            self.id2relation.update(reversed_id2relation)

            self.num_relations = 2 * num_relations
        else:
            self.num_relations = num_relations
        self.num_entities = len(self.id2entity)
        self.id2relation[self.num_relations] = 'selfloop'

        self.train_data = self._load_data(os.path.join(DataDir, dataset), "train")
        self.valid_data = self._load_data(os.path.join(DataDir, dataset), "valid")
        self.test_data = self._load_data(os.path.join(DataDir, dataset), "test")

        # add reverse event into the data set
        if add_reverse_relation:
            self.train_data = np.concatenate([self.train_data[:, :-1],
                                              np.vstack([[event[2], event[1] + num_relations, event[0], event[3]]
                                                   for event in self.train_data])], axis=0)
        seen_entities = set(self.train_data[:, 0]).union(set(self.train_data[:, 2]))
        seen_relations = set(self.train_data[:, 1])

        val_mask = [evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations
                    for evt in self.valid_data]
        self.valid_data_seen_entity = self.valid_data[val_mask]

        if add_reverse_relation:
            self.valid_data = np.concatenate([self.valid_data[:, :-1],
                                              np.vstack([[event[2], event[1] + num_relations, event[0], event[3]]
                                                   for event in self.valid_data])], axis=0)
            self.valid_data_seen_entity = np.concatenate([self.valid_data_seen_entity[:, :-1],
                                                    np.vstack([[event[2], event[1] + num_relations, event[0], event[3]]
                                                               for event in self.valid_data_seen_entity])], axis=0)

        test_mask = [evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations
                     for evt in self.test_data]
        test_mask_conjugate = ~np.array(test_mask)

        print('seen dataset proportion: ' + str(np.asarray(test_mask).sum()/len(test_mask)))
        print('unseen dataset proportion: ' + str(test_mask_conjugate.sum()/test_mask_conjugate.size))

        self.test_data_seen_entity = self.test_data[test_mask]
        self.test_data_unseen_entity = self.test_data[test_mask_conjugate]

        if add_reverse_relation:
            self.test_data = np.concatenate([self.test_data[:, :-1],
                                             np.vstack(
                                                 [[event[2], event[1] + num_relations, event[0], event[3]]
                                                  for event in self.test_data])], axis=0)
            self.test_data_seen_entity = np.concatenate([self.test_data_seen_entity[:, :-1],
                                                         np.vstack(
                                                             [[event[2], event[1] + num_relations, event[0], event[3]]
                                                              for event in self.test_data_seen_entity])], axis=0)
            self.test_data_unseen_entity = np.concatenate([self.test_data_unseen_entity[:, :-1],
                                                         np.vstack(
                                                             [[event[2], event[1] + num_relations, event[0], event[3]]
                                                              for event in self.test_data_unseen_entity])], axis=0)

        self.data = np.concatenate([self.train_data, self.valid_data, self.test_data], axis=0)
        self.timestamps = self._get_timestamps(self.data)

    def _load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, "{}.txt".format(data_type)), 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = np.array([line.split("\t") for line in data])  # only cut by "\t", not by white space.
            data = np.vstack([[int(_.strip()) for _ in line] for line in data])  # remove white space
        return data

    @staticmethod
    def _get_timestamps(data):
        timestamps = np.array(sorted(list(set([d[3] for d in data]))))
        return timestamps

    def neg_sampling_object(self, Q, dataset='train', start_time=0):
        '''
        :param Q: number of negative sampling for each real quadruple
        :param start_time: neg sampling for events since start_time (inclusive), used for warm start training
        :param dataset: indicate which data set to choose negative sampling from
        :return:
        List[List[int]]: [len(train_data), Q], list of Q negative sampling for each event in train_data
        '''
        neg_object = []
        spt_o = defaultdict(list)  # dict: (s, p, r)--> [o]
        if dataset == 'train':
            contents_dataset = self.train_data
            assert start_time < max(self.train_data[:, 3])
        elif dataset == 'valid':
            contents_dataset = self.valid_data_seen_entity
            assert start_time < max(self.valid_data_seen_entity[:, 3])
        elif dataset == 'test':
            contents_dataset = self.test_data_seen_entity
            assert start_time < max(self.test_data_seen_entity[:, 3])
        else:
            raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")

        data_after_start_time = [event for event in contents_dataset if event[3] >= start_time]
        for event in data_after_start_time:
            spt_o[(event[0], event[1], event[3])].append(event[2])
        for event in data_after_start_time:
            neg_object_one_node = []
            while True:
                candidate = np.random.choice(self.num_entities)
                if candidate not in spt_o[(event[0], event[1], event[3])]:
                    neg_object_one_node.append(
                        candidate)  # 0-th is a dummy node used to stuff the neighborhood when there is not enough nodes in the neighborhood
                if len(neg_object_one_node) == Q:
                    neg_object.append(neg_object_one_node)
                    break

        return np.stack(neg_object, axis=0)

    def _id2entity(self, dataset):
        with open(os.path.join(DataDir, dataset, "entity2id.txt"), 'r', encoding='utf-8') as f:
            mapping = f.readlines()
            mapping = [entity.strip().split("\t") for entity in mapping]
            mapping = {int(ent2idx[1].strip()): ent2idx[0].strip() for ent2idx in mapping}
        return mapping

    def _id2relation(self, dataset):
        with open(os.path.join(DataDir, dataset, "relation2id.txt"), 'r', encoding='utf-8') as f:
            mapping = f.readlines()
            mapping = [relation.strip().split("\t") for relation in mapping]
            id2relation = {}
            for rel2idx in mapping:
                id2relation[int(rel2idx[1].strip())] = rel2idx[0].strip()
        return id2relation

    def get_adj_list(self):
        '''
        adj_list for the whole dataset, including training data, validation data and test data
        :return:
        adj_list: List[List[(o(int), p(str), t(int))]], adj_list[i] is the list of (o,p,t) of events
        where entity i is the subject. Each row is sorted by timestamp of events, object and relation index
        '''
        adj_list_dict = defaultdict(list)
        for event in self.data:
            adj_list_dict[int(event[0])].append((int(event[2]), int(event[1]), int(event[3])))

        subject_index_sorted = sorted(adj_list_dict.keys())
        adj_list = [sorted(adj_list_dict[_], key=lambda x: (x[2], x[0], x[1])) for _ in subject_index_sorted]

        return adj_list

    def get_adj_dict(self):
        '''
        same as get_adj_list, but return dictionary, key is the index of subject
        :return:
        '''
        adj_dict = defaultdict(list)
        for event in self.data:
            adj_dict[int(event[0])].append((int(event[2]), int(event[1]), int(event[3])))

        for value in adj_dict.values():
            value.sort(key=lambda x: (x[2], x[0], x[1]))

        return adj_dict

    def get_spt2o(self, dataset: str):
        '''
        mapping between (s, p, t) -> list(o), i.e. values of dict are objects share the same subject, predicate and time.
        calculated for the convenience of evaluation "fil" on object prediction
        :param dataset: 'train', 'valid', 'test'
        :return:
        dict (s, p, t) -> o
        '''
        if dataset == 'train':
            events = self.train_data
        elif dataset == 'valid':
            events = self.valid_data
        elif dataset == 'test':
            events = self.test_data
        else:
            raise ValueError("invalid input {} for dataset, please input 'train', 'valid' or 'test'".format(dataset))
        spt2o = defaultdict(list)
        for event in events:
            spt2o[(event[0], event[1], event[3])].append(event[2])
        return spt2o

    def get_sp2o(self):
        '''
        get dict d which mapping between (s, p) -> list(o). More specifically, for each event in the **whole data set**,
        including training, validation and test data set, its object will be in d[(s,p)]
        it's calculated for the convenience of a looser evaluation "fil" on object prediction
        :param dataset: 'train', 'valid', 'test'
        :return:
        dict (s, p) -> o
        '''
        sp2o = defaultdict(list)
        for event in self.data:
            sp2o[(event[0], event[1])].append(event[2])
        return sp2o


class NeighborFinder:
    def __init__(self, adj, sampling=1, max_time=366 * 24, num_entities=None, weight_factor=1, time_granularity=24):
        """
        Params
        ------
        adj: list or dict, if list: adj[i] is the list of all (o,p,t) for entity i, if dict: adj[i] is the list of all (o,p,t)
        sampling: sample strategy from neighborhood: -1: whole neighbor, 0: uniform, 1: first N, 2: last N, 3: time difference weighted sampling
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i][:,0]
        off_set_t_l: node_idx_l[off_set_l[i]:off_set_l[i + 1]][:off_set_t_l[i][cut_time/time_granularity]] --> object of entity i that happen before cut time
        num_entities: number of entities, if adj is dict it cannot be None
        weight_factor: if sampling==3, use weight_factor to scale the time difference
        """

        self.time_granularity = time_granularity
        node_idx_l, node_ts_l, edge_idx_l, off_set_l, off_set_t_l = self.init_off_set(adj, max_time, num_entities)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l
        self.off_set_t_l = off_set_t_l

        self.sampling = sampling
        self.weight_factor = weight_factor

    def init_off_set(self, adj, max_time, num_entities):
        """
        for events with entity of index i being subject:
        node_idx_l[off_set_l[i]:off_set_l[i+1]] is the list of object index
        node_ts_l[off_set_l[i]:off_set_l[i+1]] is the list of timestamp
        edge_idx_l[off_set_l[i]:off_set_l[i+1]] is the list of relation
        ordered by (ts, ent, rel) ascending
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        off_set_t_l = []

        if isinstance(adj, list):
            for i in range(len(adj)):
                assert len(adj) == num_entities
                curr = adj[i]
                curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                curr_ts = [x[2] for x in curr]
                n_ts_l.extend(curr_ts)

                off_set_l.append(len(n_idx_l))
                off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in range(0, max_time+1, self.time_granularity)])# max_time+1 so we have max_time
        elif isinstance(adj, dict):
            for i in range(num_entities):
                curr = adj.get(i, [])
                curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                curr_ts = [x[2] for x in curr]
                n_ts_l.extend(curr_ts)

                off_set_l.append(len(n_idx_l))
                off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in range(0, max_time+1, self.time_granularity)])# max_time+1 so we have max_time

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l, off_set_t_l

    def get_temporal_degree(self, src_idx_l, cut_time_l):
        """
        return how many neighbros exist for each (src, ts)
        :param src_idx_l:
        :param cut_time_l:
        :return:
        """
        assert (len(src_idx_l) == len(cut_time_l))

        temp_degree = []
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            temp_degree.append(self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)])  # every timestamp in neighbors_ts[:mid] is smaller than cut_time
        return np.array(temp_degree)

    def find_before(self, src_idx, cut_time):
        neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        mid = np.searchsorted(neighbors_ts, cut_time)
        ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]
        return ngh_idx, ngh_eidx, ngh_ts

    def get_temporal_neighbor_v2(self, src_idx_l, cut_time_l, query_time_l, num_neighbors=20):
        """
        temporal neighbors are not limited to be drawn from events happen before cut_time,
        but are extended to be drawn from all events that happen before query time
        More specifically, for each query we have (sub_q, rel_q, ?, t_q). By each step, for
        every node, i.e. entity-timestamp pair (e_i, t_i), we looked for such entity-timestamp
        pair (e, t) that (e_i, some_relation, e, t) exists. By first step, (e_i, t_i) == (sub_q, t_q)
        where t < t_q is the restriction (rather than t<t_0)
        Arguments:
            src_idx_l {numpy.array, 1d} -- entity index
            cut_time_l {numpy.array, 1d} -- timestamp of events
            query_time_l {numpy.array, 1d} -- timestamp of query

        Keyword Arguments:
            num_neighbors {int} -- [number of neighbors for each node] (default: {20})
        """
        assert (len(src_idx_l) == len(cut_time_l))
        assert (len(src_idx_l) == len(query_time_l))
        assert all([cut_time <= query_time for cut_time, query_time in list(zip(cut_time_l, query_time_l))])
        assert (num_neighbors % 2 == 0)

        out_ngh_node_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time, query_time) in enumerate(zip(src_idx_l, cut_time_l, query_time_l)):
            neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            mid = self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)]
            end = self.off_set_t_l[src_idx][int(query_time / self.time_granularity)]
            # every timestamp in neighbors_ts[:mid] is smaller than cut_time
            ngh_idx_before, ngh_eidx_before, ngh_ts_before = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]
            # every timestamp in neighbors_ts[mid:end] is bigger than cut_time and smaller than query_time
            ngh_idx_after, ngh_eidx_after, ngh_ts_after = neighbors_idx[mid:end], neighbors_e_idx[mid:end], neighbors_ts[mid:end]

            # choose events happen closest in time
            half_num_neighbors = num_neighbors//2
            ngh_ts_before = ngh_ts_before[-half_num_neighbors:]
            ngh_idx_before = ngh_idx_before[-half_num_neighbors:]
            ngh_eidx_before = ngh_eidx_before[-half_num_neighbors:]

            out_ngh_node_batch[i, half_num_neighbors - len(ngh_idx_before):half_num_neighbors] = ngh_idx_before
            out_ngh_t_batch[i, half_num_neighbors - len(ngh_ts_before):half_num_neighbors] = ngh_ts_before
            out_ngh_eidx_batch[i, half_num_neighbors - len(ngh_eidx_before):half_num_neighbors] = ngh_eidx_before

            ngh_ts_after = ngh_ts_after[:half_num_neighbors]
            ngh_idx_after = ngh_idx_after[:half_num_neighbors]
            ngh_eidx_after = ngh_eidx_after[:half_num_neighbors]

            out_ngh_node_batch[i, half_num_neighbors:len(ngh_eidx_after) + half_num_neighbors] = ngh_idx_after
            out_ngh_t_batch[i, half_num_neighbors: len(ngh_ts_after) + half_num_neighbors] = ngh_ts_after
            out_ngh_eidx_batch[i, half_num_neighbors: len(ngh_eidx_after) + half_num_neighbors] = ngh_eidx_after

        out_ngh_query_t_batch = np.repeat(np.repeat(query_time_l[:, np.newaxis], num_neighbors, axis=1))

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_query_t_batch

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        each entity has exact num_neighbors neighbors, neighbors are sampled according to sample strategy
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int, ignored if sampling==-1,
        return:
        out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch: sorted by out_ngh_t_batch
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_eidx_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)

        if self.sampling == -1:
            full_ngh_node = []
            full_ngh_t = []
            full_ngh_edge = []
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            mid = self.off_set_t_l[src_idx][
                int(cut_time / self.time_granularity)]  # every timestamp in neighbors_ts[:mid] is smaller than cut_time
            ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]

            if len(ngh_idx) > 0:
                if self.sampling == 0:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    sampled_idx = np.sort(sampled_idx)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                elif self.sampling == 1:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                elif self.sampling == 2:
                    ngh_ts = ngh_ts[-num_neighbors:]
                    ngh_idx = ngh_idx[-num_neighbors:]
                    ngh_eidx = ngh_eidx[-num_neighbors:]
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                elif self.sampling == 3:
                    delta_t = (ngh_ts - cut_time)/(self.time_granularity*self.weight_factor)
                    weights = np.exp(delta_t) + 1e-9
                    weights = weights / sum(weights)

                    if len(ngh_idx) >= num_neighbors:
                        sampled_idx = np.random.choice(len(ngh_idx), num_neighbors, replace=False, p=weights)
                    else:
                        sampled_idx = np.random.choice(len(ngh_idx), len(ngh_idx), replace=False, p=weights)

                    sampled_idx = np.sort(sampled_idx)
                    out_ngh_node_batch[i, num_neighbors - len(sampled_idx):] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, num_neighbors - len(sampled_idx):] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, num_neighbors - len(sampled_idx):] = ngh_eidx[sampled_idx]
                elif self.sampling == 4:
                    weights = (ngh_ts + 1) / sum(ngh_ts + 1)

                    if len(ngh_idx) >= num_neighbors:
                        sampled_idx = np.random.choice(len(ngh_idx), num_neighbors, replace=False, p=weights)
                    else:
                        sampled_idx = np.random.choice(len(ngh_idx), len(ngh_idx), replace=False, p=weights)

                    sampled_idx = np.sort(sampled_idx)
                    out_ngh_node_batch[i, num_neighbors - len(sampled_idx):] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, num_neighbors - len(sampled_idx):] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, num_neighbors - len(sampled_idx):] = ngh_eidx[sampled_idx]

                elif self.sampling == -1: # use whole neighborhood
                    full_ngh_node.append(ngh_idx[-300:])
                    full_ngh_t.append(ngh_ts[-300:])
                    full_ngh_edge.append(ngh_eidx[-300:])
                else:
                    raise ValueError("invalid input for sampling")

        if self.sampling == -1:
            max_num_neighbors = max(map(len, full_ngh_edge))
            out_ngh_node_batch = -np.ones((len(src_idx_l), max_num_neighbors)).astype(np.int32)
            out_ngh_t_batch = np.zeros((len(src_idx_l), max_num_neighbors)).astype(np.int32)
            out_ngh_eidx_batch = -np.ones((len(src_idx_l), max_num_neighbors)).astype(np.int32)
            for i in range(len(full_ngh_node)):
                out_ngh_node_batch[i, max_num_neighbors-len(full_ngh_node[i]):] = full_ngh_node[i]
                out_ngh_eidx_batch[i, max_num_neighbors-len(full_ngh_edge[i]):] = full_ngh_edge[i]
                out_ngh_t_batch[i, max_num_neighbors-len(full_ngh_t[i]):] = full_ngh_t[i]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def get_neighbor_subgraph(self, src_idx_l, cut_time_l, level=2, num_neighbors=20):
        Gs = [nx.Graph() for _ in range(len(src_idx_l))]
        for i, G in enumerate(Gs):
            G.add_node((src_idx_l[i], None, cut_time_l[i]), rel=None, time=cut_time_l[i])

        def get_neighbors_recursive(graph_index_l, src_idx_l, rel_idx_l, cut_time_l, level, num_neighbors):
            if level == 0:
                return
            else:
                src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.get_temporal_neighbor(
                    src_idx_l,
                    cut_time_l,
                    num_neighbors=num_neighbors)

                for batch_idx in range(len(src_idx_l)):
                    ngh_nodes = src_ngh_node_batch[batch_idx]
                    ngh_edges = src_ngh_eidx_batch[batch_idx]
                    ngh_ts = src_ngh_t_batch[batch_idx]

                    Gs[graph_index_l[batch_idx]].add_nodes_from(
                        [((node, rel, t), {'rel': rel, 'time': t}) for node, rel, t in
                         list(zip(ngh_nodes, ngh_edges, ngh_ts))])
                    Gs[graph_index_l[batch_idx]].add_edges_from([((src_idx_l[batch_idx], rel_idx_l[batch_idx], cut_time_l[batch_idx]),
                                                                  (node, edge, t))
                                                                 for node, edge, t in list(zip(ngh_nodes, ngh_edges, ngh_ts))])

                    get_neighbors_recursive(np.repeat(graph_index_l[batch_idx],
                                                      len(ngh_nodes)), ngh_nodes, ngh_edges, ngh_ts, level - 1, num_neighbors)

        get_neighbors_recursive(np.arange(len(src_idx_l)), src_idx_l, [None for _ in src_idx_l], cut_time_l, level, num_neighbors)
        return Gs


class Measure:
    '''
    Evaluation of link prediction.
    raw: Given (s, o, t), measurement based on the rank of a true (s, r, o, t) in all possible (s, r, o, t)
    fil: Given (s, o, t), measurement based on the rank of a true (s, r, o, t) in all (s, r, o, t) that don't happen.
    '''

    def __init__(self):
        '''
        mr: mean rank
        mrr: mean reciprocal rank
        '''
        self.hit1 = {"raw": 0.0, "fil": 0.0}
        self.hit3 = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr = {"raw": 0.0, "fil": 0.0}
        self.mr = {"raw": 0.0, "fil": 0.0}

    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil] += rank
        self.mrr[raw_or_fil] += (1.0 / rank)

    def batch_update(self, rank_l, raw_or_fil):
        self.hit1[raw_or_fil] += np.sum(rank_l == 1)
        self.hit3[raw_or_fil] += np.sum(rank_l <= 3)
        self.hit10[raw_or_fil] += np.sum(rank_l <= 10)
        self.mr[raw_or_fil] += np.sum(rank_l)
        self.mrr[raw_or_fil] += np.reciprocal(rank_l)

    def normalize(self, num_facts):
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil] /= num_facts
            self.hit3[raw_or_fil] /= num_facts
            self.hit10[raw_or_fil] /= num_facts
            self.mr[raw_or_fil] /= num_facts
            self.mrr[raw_or_fil] /= num_facts

    def print_(self):
        for raw_or_fil in ["raw", "fil"]:
            print(raw_or_fil.title() + " setting:")
            print("\tHit@1 =", self.hit1[raw_or_fil])
            print("\tHit@3 =", self.hit3[raw_or_fil])
            print("\tHit@10 =", self.hit10[raw_or_fil])
            print("\tMR =", self.mr[raw_or_fil])
            print("\tMRR =", self.mrr[raw_or_fil])


def get_git_version_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])[:-1].decode("utf-8")

def get_git_description_last_commit():
    return subprocess.check_output(['git', 'log', '-2', '--pretty=%B']).decode('utf-8')


def save_config(args, dir: str):
    args_dict = vars(args)
    git_hash = get_git_version_short_hash()
    git_comment = get_git_description_last_commit()
    args_dict['git_hash'] = '\t'.join([git_hash, git_comment])
    if not os.path.exists(os.path.join(dir, 'config.json')):
        with open(os.path.join(dir, 'config.json'), 'w') as fp:
            json.dump(args_dict, fp)
            print("Log configuration under {}".format(os.path.join(dir, 'config.json')))


def get_segment_ids(x):
    """ x: (np.array) d0 x 2, sorted
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    y = (x[1:] == x[:-1]).astype('uint8')
    return np.concatenate([np.array([0], dtype='int32'),
                           np.cumsum(1 - y[:, 0] * y[:, 1], dtype='int32')])


def load_checkpoint(checkpoint_dir, device='cpu', args=None):
    if os.path.isfile(checkpoint_dir):
        print("=> loading checkpoint '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location=torch.device(device))
        start_epoch = checkpoint['epoch']

        if 'args' in checkpoint.keys():
            # earlier checkpoint doesn't contain hyperparameters, i.e. 'args'
            loaded_args = checkpoint['args']
            if args and set(vars(args).keys()) != set(vars(loaded_args).keys()):
                print("ATTENTION!!Mismatch between current arguments and saved arguments in checkpoint")
            args = loaded_args
            print("use args in checkpoint:", args)
        else:
            assert args is not None
            print("use args from command line:", args)
        contents = Data(dataset=args.dataset, add_reverse_relation=args.add_reverse)

        adj = contents.get_adj_dict()
        max_time = max(contents.data[:, 3])

        if 'yago' in args.dataset.lower():
            time_granularity = 1
        elif 'icews' in args.dataset.lower():
            time_granularity = 24

        nf = NeighborFinder(adj, sampling=args.sampling, max_time=max_time, num_entities=len(contents.id2entity),
                            weight_factor=args.weight_factor, time_granularity=time_granularity)
        kwargs = vars(args)
        kwargs['device'] = device
        model = xERTE(nf, contents.num_entities, contents.num_relations, use_time_embedding=not args.no_time_embedding, **kwargs)
        # move a model to GPU before constructing an optimizer, http://pytorch.org/docs/master/optim.html
        model.to(device)
        model.entity_raw_embed.cpu()
        model.relation_raw_embed.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch']))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(checkpoint_dir))

    return model, optimizer, start_epoch, contents, args


def new_checkpoint(save_dir, struct_time):
    checkpoint_dir = 'checkpoints_{}_{}_{}_{}_{}_{}'.format(
        struct_time.tm_year,
        struct_time.tm_mon,
        struct_time.tm_mday,
        struct_time.tm_hour,
        struct_time.tm_min,
        struct_time.tm_sec)
    CHECKPOINT_PATH = os.path.join(save_dir, 'Checkpoints', checkpoint_dir)
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, mode=0o770)
    return checkpoint_dir, CHECKPOINT_PATH
