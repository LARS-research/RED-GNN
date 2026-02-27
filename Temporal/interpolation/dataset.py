import torch
from torch.utils import data
from copy import deepcopy
import numpy as np

from graph import TemporalGraph


class Example(object):
    """Defines each triple in TKG"""
    def __init__(self, triple, entity_vocab, relation_vocab, time_vocab, example_idx):
        self.head_idx = entity_vocab(triple[0])
        self.relation_idx = relation_vocab(triple[1])
        self.tail_idx = entity_vocab(triple[2])
        self.time_idx = time_vocab(triple[3])
        self.example_idx = example_idx

        self.graph = None


class TKGDataset(data.Dataset):
    """Temporal KG Dataset Class"""
    def __init__(self, example_list, kg, device):
        self.example_list = example_list
        self.kg = kg
        self.device = device
        self.graph = self.kg.graph

    def __iter__(self):
        return iter(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        return example

    def __len__(self):
        return len(self.example_list)

    def collate(self, batched_examples):
        batch_heads, batch_relations, batch_tails, batch_times, batch_graph, batch_ex_indices = [], [], [], [], [], []
        for example in batched_examples:
            batch_heads.append(example.head_idx)
            batch_relations.append(example.relation_idx)
            batch_tails.append(example.tail_idx)
            batch_times.append(example.time_idx)
            batch_ex_indices.append(example.example_idx)
        

        return {
            "head": torch.tensor(batch_heads),
            "relation": torch.tensor(batch_relations),
            "tail": torch.tensor(batch_tails),
            "time": torch.tensor(batch_times),
            "example_idx": torch.tensor(batch_ex_indices),
            # "graph": graph
        }


def get_datasets(filenames, device):
    KG = TemporalGraph(filenames[0], device)
    datasets = []

    for fname in filenames:
        triples = open(fname, 'r').read().lower().splitlines()
        triples = list(map(lambda x: x.split("\t"), triples))

        # for i in range(len(triples)):
        #     triples[i][3] = triples[0][3]
        # triples = [tuple(i) for i in triples]
        # triples = list(set(triples))
        # triples = [list(i) for i in triples]

        example_list = []
        for i, triple in enumerate(triples):
            example_list.append(Example(triple, KG.entity_vocab, KG.relation_vocab, KG.time_vocab, i))

        datasets.append(TKGDataset(example_list, KG, device))

    return datasets