import os
from collections.abc import Iterable
from typing import Tuple
import subprocess
import logging
import torch
import numpy as np
import json
_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_SCHEDULER_STATE_DICT = "scheduler_state_dict"
_EPOCH = "epoch"
_COUNT = "global_count"


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    """Loads checkpoint file"""
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])

    if optimizer:
        optimizer.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    if scheduler:
        scheduler.load_state_dict(checkpoint[_SCHEDULER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    global_count = checkpoint[_COUNT]

    return start_epoch_id, global_count


def save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch_id, global_count, metric):
    """Save state to checkpoint file"""
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optimizer.state_dict(),
        _SCHEDULER_STATE_DICT: scheduler.state_dict(),
        _EPOCH: epoch_id,
        _COUNT: global_count,
    }, os.path.join(ckpt_path, f"{metric:.5}.{epoch_id}.tar"))


def hits_at_k(predicted_prob, target_indices, k):
    """
    :param predicted_prob: (batch_size, num_nodes)
    :param target_indices: (batch_size, )
    :param k: number of nodes to decode
    :return: number of correct inferences
    """
    topk = torch.topk(predicted_prob, dim=1, k=k)[1]

    return torch.sum(topk == target_indices.unsqueeze(1)).item()


class Vocab(object):
    """Entity / Relation / Timestamp Vocabulary Class"""
    def __init__(self, max_vocab=2**31, min_freq=-1, sp=None):
        if sp is None:
            sp = ['_PAD', '_UNK']
        self.itos = []
        self.stoi = {}
        self.freq = {}
        self.max_vocab, self.min_freq, self.sp = max_vocab, min_freq, sp

    def __len__(self):
        return len(self.itos)

    def __str__(self):
        return 'Total ' + str(len(self.itos)) + str(self.itos[:10])

    def update(self, token):
        if isinstance(token, Iterable):
            for t in token:
                self.freq[t] = self.freq.get(t, 0) + 1
        else:
            self.freq[token] = self.freq.get(token, 0) + 1

    def build(self, sort_key="freq"):
        assert len(self.itos) == 0 and len(self.stoi) == 0, "Build should only be called for initialization."
        self.itos.extend(self.sp)

        freq = sorted(self.freq.items(), key=lambda x: x[1] if sort_key == "freq" else x[0],
                      reverse=(sort_key == "freq"))

        for k, v in freq:
            if len(self.itos) < self.max_vocab and k not in self.sp and v >= self.min_freq:
                self.itos.append(k)
        self.stoi.update(list(zip(self.itos, range(len(self.itos)))))

    def __call__(self, x):
        if isinstance(x, int):
            return self.itos[x]
        else:
            return self.stoi.get(x, self.stoi['_UNK'])


def pad(tensor_list, pad_idx) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads list of tensors with maximal length, and return stacked tensor / lengths"""
    lens = torch.Tensor([x.size(0) for x in tensor_list]).long()
    max_len = max([x.size(0) for x in tensor_list])

    return torch.stack(
        [torch.cat([x, torch.full([max_len-len(x)] + list(x.shape[1:]), pad_idx).type_as(x)], 0) for x in tensor_list],
        dim=0), lens

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            #proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0,len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i

def save_result(args, valid_hits1):
    dict_res = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'epoch': args.epoch,
        'grad_clip': str(args.grad_clip),
        'dataset': args.dataset,
        'decay_rate': args.decay_rate,
        'hidden_dim': args.hidden_dim,
        'attn_dim': args.attn_dim,
        'n_layer': args.n_layer,
        'dropout': args.dropout,
        'hit1': valid_hits1,
        'act': args.act,
    }
    json.dump(dict_res, open(f'result/{args.time}.json', 'w'))