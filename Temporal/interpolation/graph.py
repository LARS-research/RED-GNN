import torch
from util import Vocab
import numpy as np
import os


class TemporalGraph:
    """Temporal Graph Container Class"""
    def __init__(self, train_path, device):
        self.device = device

        print(os.getcwd())

        with open(train_path, 'r') as f:
            lines = f.read().lower().splitlines()
            lines = map(lambda x: x.split("\t"), lines)


            head_list, relation_list, tail_list, time_list = tuple(zip(*lines))
            self.entity_vocab = Vocab()
            self.relation_vocab = Vocab()
            self.time_vocab = Vocab()
            self.entity_vocab.update(head_list + tail_list)
            self.relation_vocab.update(list(relation_list) + ['idd'])
            if 'wiki' not in train_path:
                self.time_vocab.update(list(time_list) + ['2020-01-01'])
            else:
                print('wikidata11k')
                self.time_vocab.update(list(time_list) + ['2050'])
            self.entity_vocab.build()
            self.relation_vocab.build()
            self.time_vocab.build(sort_key="time")

            # add idd tuples 
            # idd relation is at the end of graph and not used as query
            unique_entity = list(set(head_list + tail_list))
            head_list = list(head_list) + unique_entity
            tail_list = list(tail_list) + unique_entity
            relation_list = list(relation_list) + ['idd' for _ in unique_entity]
            time_list = list(time_list) + ['2020-01-01' for _ in unique_entity]

            head_list = list(map(lambda x: self.entity_vocab(x), head_list))
            relation_list = list(map(lambda x: self.relation_vocab(x), relation_list))
            tail_list = list(map(lambda x: self.entity_vocab(x), tail_list))
            time_list = list(map(lambda x: self.time_vocab(x), time_list))


            # time_list = list(map(lambda x: 1, time_list))
            self.graph = np.array([head_list, relation_list, tail_list, time_list]).T
            # self.graph = np.unique(self.graph, axis=0)

        # self.graph = dgl.DGLGraph(multigraph=True)
        # self.graph.add_nodes(len(self.entity_vocab))
        # self.graph.add_edges(head_list, tail_list)
        # self.graph.ndata['node_idx'] = torch.arange(self.graph.number_of_nodes())
        # self.graph.edata['relation_type'] = torch.tensor(relation_list)
        # self.graph.edata['time'] = torch.tensor(time_list)

        print("Graph prepared.")
