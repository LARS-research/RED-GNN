import imp
import os
import sys
import json
import subprocess
from collections import defaultdict
import time
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import rankdata


DataDir = os.path.join(os.path.dirname(__file__), 'data')
class Data_v2:
    def __init__(self, dataset=None, add_reverse_relation=True) -> None:
        self.id2entity = self._id2entity(dataset=dataset)
        self.id2relation = self._id2relation(dataset=dataset)
        self.num_relations = len(self.id2relation)
        self.num_entities = len(self.id2entity)
        reversed_id2relation = {}
        for ind, rel in self.id2relation.items():
            reversed_id2relation[ind + self.num_relations] = 'Reversed ' + rel
        self.id2relation.update(reversed_id2relation)

        # self.num_entities = len(self.id2entity)
        self.id2relation[2 * self.num_relations + 1] = 'selfloop'
        
        self.train_data = self._load_data(os.path.join(DataDir, dataset), "train")
        self.valid_data = self._load_data(os.path.join(DataDir, dataset), "valid")
        self.test_data = self._load_data(os.path.join(DataDir, dataset), "test")
        

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
    
    def _load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, "{}.txt".format(data_type)), 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = np.array([line.split("\t") for line in data])  # only cut by "\t", not by white space.
            data = np.vstack([[int(_.strip()) for _ in line] for line in data])  # remove white space
            data = np.unique(data[:,:3], axis=0)
            data_inverse = data[:, [2, 1, 0]]
            data_inverse[:, 1] += self.num_relations
            data_identity = np.array([[i, 2*self.num_relations+1, i] for i in range(self.num_entities)])
            data = np.concatenate([data, data_inverse, data_identity])
        return data
    
    def get_adj(self, data_type='train'):
        if data_type == 'train':
            res = {}
            for i in self.train_data:
                if i[0] not in res.keys():
                    res[i[0]] = [i]
                else:
                    res[i[0]].append(i)
            return res
        elif data_type == 'valid':
            res = {}
            for i in self.valid_data:
                if i[0] not in res.keys():
                    res[i[0]] = [i]
                else:
                    res[i[0]].append(i)
            return res
        elif data_type == 'test':
            res = {}
            for i in self.test_data:
                if i[0] not in res.keys():
                    res[i[0]] = [i]
                else:
                    res[i[0]].append(i)
            return res
        


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

def prepare_inputs(contents, dataset='train', start_time=0, tc=None):
    '''
    :param tc: time recorder
    :param contents: instance of Data object
    :param num_neg_sampling: how many negtive sampling of objects for each event
    :param start_time: neg sampling for events since start_time (inclusive)
    :param dataset: 'train', 'valid', 'test'
    :return:
    events concatenated with negative sampling
    '''
    t_start = time.time()
    if dataset == 'train':
        contents_dataset = contents.train_data
        # assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'valid':
        contents_dataset = contents.valid_data
        # assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'test':
        contents_dataset = contents.test_data
        # assert start_time < max(contents_dataset[:, 3])
    else:
        raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")
    events = np.vstack([np.array(event) for event in contents_dataset])
    if None:
        tc['data']['load_data'] += time.time() - t_start
    return events

def find_neighbours(G, entity, ts):
    pass

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True)
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10

def cal_mrr(scores, labels):
    full_rank = rankdata(-scores, method='average', axis=1)
    rank = np.multiply(full_rank, labels)
    rank_batch = np.sum(rank, axis=1) / np.sum(labels, 1)
    return rank_batch.tolist()