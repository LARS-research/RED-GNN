import os
# the mode argument of the os.makedirs function may be ignored on some systems
# umask (user file-creation mode mask) specify the default denial value of variable mode, 
# which means if this value is passed to makedirs function,  
# it will be ignored and a folder/file with d_________ will be created 
# we can either set the umask or specify mode in makedirs
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"

# oldmask = os.umask(0o770)

import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
# import setproctitle
from main import Options

# setproctitle.setproctitle('TKG@QiuHaiquan')

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, save_config, load_checkpoint
import local_config
from segment import *
from database_op import DBDriver

save_dir = local_config.save_dir

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class Options:
    pass

def reset_time_cost():
    return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float),
            'data': defaultdict(float)}


def str_time_cost(tc):
    if tc is not None:
        data_tc = ', '.join('data.{} {:3f}'.format(k, v) for k, v in tc['data'].items())
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in tc['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in tc['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in tc['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def prepare_inputs(contents, whole_or_seen, dataset='train', start_time=0, tc=None):
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
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'valid':
        if whole_or_seen == 'whole':
            contents_dataset = contents.valid_data
        else:
            contents_dataset = contents.valid_data_seen_entity
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'test':
        if whole_or_seen == 'whole':
            contents_dataset = contents.test_data
        elif whole_or_seen == 'seen':
            contents_dataset = contents.test_data_seen_entity
        elif whole_or_seen == 'unseen':
            contents_dataset = contents.test_data_unseen_entity
        else:
            raise NotImplemented
        assert start_time < max(contents_dataset[:, 3])
    else:
        raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")
    events = np.vstack([np.array(event) for event in contents_dataset if event[3] >= start_time])
    #     neg_obj_idx = contents.neg_sampling_object(num_neg_sampling, dataset=dataset, start_time=start_time)
    if args.timer:
        tc['data']['load_data'] += time.time() - t_start
    #     return np.concatenate([events, neg_obj_idx], axis=1)
    return events


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.src_idx = np.array(transposed_data[0], dtype=np.int32)
        self.rel_idx = np.array(transposed_data[1], dtype=np.int32)
        self.target_idx = np.array(transposed_data[2], dtype=np.int32)
        self.ts = np.array(transposed_data[3], dtype=np.int32)
        self.neg_idx = np.array(transposed_data[4:-1], dtype=np.int32).T
        self.event_idx = np.array(transposed_data[-1], dtype=np.int32)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_idx = self.src_idx.pin_memory()
        self.rel_idx = self.rel_idx.pin_memory()
        self.target_idx = self.target_idx.pin_memory()
        self.neg_idx = self.neg_idx.pin_memory()
        self.ts = self.ts.pin_memory()
        self.event_idx = self.event_idx.pin_memory()

        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def segment_topk(t, segment_idx, k, sorted=False):
    """
    compute topk along segments of a tensor
    params:
        t: Tensor, 1d, dtype=torch.float32
        segment_idx: numpy.array, 1d, dtype=numpy.int32, sorted
        k: k largest values
    return:
        values[i]: Tensor of topk of segment i
        indices[i]: numpy.array of position of topk elements of segment i in original Tensor t
    """
    mask = segment_idx[1:] != segment_idx[:-1]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(segment_idx))[mask],
                              np.array([len(segment_idx)])])
    values = []
    indices = []
    for s, e in zip(key_idx[:-1], key_idx[1:]):
        if e - s < k:
            if sorted:
                sorted_value, sorted_indices = torch.sort(t[s:e], descending=True)
                values.append(sorted_value)
                indices.append(s + sorted_indices.cpu().numpy())
            else:
                values.append(t[s:e])
                indices.append(np.arange(s, e))
        else:
            segment_values, segment_indices = torch.topk(t[s:e], k, sorted=sorted)
            values.append(segment_values)
            indices.append(s + segment_indices.cpu().numpy())
    return values, indices


def segment_rank(t, entities, target_idx_l):
    """
    compute rank of ground truth (target_idx_l) in prediction according to score, i.e. t
    :param t: prediction score
    :param entities: 2-d numpy array, (segment_idx, entity_idx)
    :param target_idx_l: 1-d numpy array, (batch_size, )
    :return:
    """
    mask = entities[1:, 0] != entities[:-1, 0]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(entities))[mask],
                              np.array([len(entities)])])
    rank = []
    found_mask = []
    for i, (s, e) in enumerate(zip(key_idx[:-1], key_idx[1:])):
        arg_target = np.nonzero(entities[s:e, 1] == target_idx_l[i])[0]
        if arg_target.size > 0:
            found_mask.append(True)
            rank.append(torch.sum(t[s:e] > t[s:e][torch.from_numpy(arg_target)]).item() + 1)
        else:
            found_mask.append(False)
            rank.append(1e9)  # MINERVA set rank to +inf if not in path, we follow this scheme
    return np.array(rank), found_mask


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='specify data set')
parser.add_argument('--whole_or_seen', type=str, default='whole', choices=['whole', 'seen', 'unseen'], help='test on the whole set or only on seen entities.')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
parser.add_argument('--sampling', type=int, default=3,
                    help='strategy to sample neighbors, 0: uniform, 1: first num_neighbors, 2: last num_neighbors')
parser.add_argument('--DP_num_neighbors', type=int, default=15, help='number of neighbors sampled for sampling horizon')
parser.add_argument('--max_attended_edges', type=int, default=40, help='max number of nodes in attending from horizon')
parser.add_argument('--load_checkpoint', type=str, default=None, help='train from checkpoints')
parser.add_argument('--timer', action='store_true', default=None, help='set to profile time consumption for some func')
parser.add_argument('--debug', action='store_true', default=None, help='in debug mode, checkpoint will not be saved')
parser.add_argument('--sqlite', action='store_true', default=None, help='save information to sqlite')
parser.add_argument('--mongo', action='store_true', default=None, help='save information to mongoDB')
parser.add_argument('--gradient_iters_per_update', type=int, default=1,
                    help='gradient accumulation, update parameters every N iterations, default 1. set when GPU memo is small')
parser.add_argument('--loss_fn', type=str, default='BCE', choices=['BCE', 'CE'])
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    time_cost = None
    if args.timer:
        time_cost = reset_time_cost()
    whole_or_seen = args.whole_or_seen
    eval_batch_size = args.batch_size
    if args.mongo or args.sqlite:
        dbDriver = DBDriver(useMongo=True, useSqlite=args.sqlite, MongoServerIP=local_config.MongoServer,
                        sqlite_dir=os.path.join(save_dir, 'tKGR.db'))

    checkpoint = args.load_checkpoint
    DP_num_neighbors = args.DP_num_neighbors
    max_attended_edges = args.max_attended_edges
    if args.load_checkpoint is None:
        raise ValueError("please specify checkpoint")
    else:
        # args will be overwritten here
        model, optimizer, start_epoch, contents, args = load_checkpoint(
            os.path.join(save_dir, 'Checkpoints', args.load_checkpoint), device)
        args.DP_num_neighbors = DP_num_neighbors
        args.max_attended_edges = max_attended_edges
        sp2o = contents.get_sp2o()
        test_spt2o = contents.get_spt2o('test')
    print(args)

    hit_1 = hit_3 = hit_10 = 0
    hit_1_fil = hit_3_fil = hit_10_fil = 0
    hit_1_fil_t = hit_3_fil_t = hit_10_fil_t = 0
    found_cnt = 0
    MR_total = 0
    MR_found = 0
    MRR_total = 0
    MRR_found = 0
    MR_total_fil = 0
    MR_total_fil_t = 0
    MRR_total_fil = 0
    MRR_total_fil_t = 0
    num_query = 0
    mean_degree = 0
    mean_degree_found = 0

    test_inputs = prepare_inputs(contents, whole_or_seen, dataset='test', tc=time_cost)
    test_data_loader = DataLoader(test_inputs, batch_size=eval_batch_size, collate_fn=collate_wrapper,
                                  pin_memory=False, shuffle=True)

    print("Start Evaluation")
    for batch_ndx, sample in enumerate(tqdm(test_data_loader)):
        model.eval()

        src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
        num_query += len(src_idx_l)
        # degree_batch = model.ngh_finder.get_temporal_degree(src_idx_l, cut_time_l)
        # mean_degree += sum(degree_batch)

        score, (entity_att_score, entities) = model(sample)
        target_idx_l = torch.from_numpy(sample.target_idx).type(torch.LongTensor).to(args.device)

        predicted_prob = F.softmax(score, dim=1)

        loss = F.nll_loss(torch.log(predicted_prob + 1e-12), target_idx_l)

        # loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size,
        #                   args.gradient_iters_per_update, args.loss_fn)

        target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = segment_rank_fil(entity_att_score,
                                                                                             entities,
                                                                                             target_idx_l.cpu().numpy(),
                                                                                             sp2o,
                                                                                             test_spt2o,
                                                                                             src_idx_l,
                                                                                             rel_idx_l,
                                                                                             cut_time_l)
        # mean_degree_found += 0
        hit_1 += np.sum(target_rank_l == 1)
        hit_3 += np.sum(target_rank_l <= 3)
        hit_10 += np.sum(target_rank_l <= 10)
        hit_1_fil += np.sum(target_rank_fil_l <= 1)  # target_rank_fil_l has dtype float
        hit_3_fil += np.sum(target_rank_fil_l <= 3)
        hit_10_fil += np.sum(target_rank_fil_l <= 10)
        hit_1_fil_t += np.sum(target_rank_fil_t_l <= 1)  # target_rank_fil_t_l has dtype float
        hit_3_fil_t += np.sum(target_rank_fil_t_l <= 3)
        hit_10_fil_t += np.sum(target_rank_fil_t_l <= 10)
        found_cnt += np.sum(found_mask)
        MR_total += np.sum(target_rank_l)
        MR_found += len(found_mask) and np.sum(
            target_rank_l[found_mask])  # if no subgraph contains ground truch, MR_found = 0 for this batch
        MRR_total += np.sum(1 / target_rank_l)
        MRR_found += len(found_mask) and np.sum(
            1 / target_rank_l[found_mask])  # if no subgraph contains ground truth, MRR_found = 0 for this batch
        MR_total_fil += np.sum(target_rank_fil_l)
        MR_total_fil_t += np.sum(target_rank_fil_t_l)
        MRR_total_fil += np.sum(1 / target_rank_fil_l)
        MRR_total_fil_t += np.sum(1 / target_rank_fil_t_l)
        # if batch_ndx == 100:
        #     break
    print(
        "Filtered performance (time dependent): Hits@1: {}, Hits@3: {}, Hits@10: {}, MRR: {}".format(
            hit_1_fil_t / num_query,
            hit_3_fil_t / num_query,
            hit_10_fil_t / num_query,
            MRR_total_fil_t / num_query))
    print(
        "Filtered performance (time independent): Hits@1: {}, Hits@3: {}, Hits@10: {}, MRR: {}".format(
            hit_1_fil / num_query,
            hit_3_fil / num_query,
            hit_10_fil / num_query,
            MRR_total_fil / num_query))
    print(
        "Raw performance: Hits@1: {}, Hits@3: {}, Hits@10: {}, Hits@Inf: {}, MR: {}, MRR: {}, degree: {}".format(
            hit_1 / num_query,
            hit_3 / num_query,
            hit_10 / num_query,
            found_cnt / num_query,
            MR_total / num_query,
            MRR_total / num_query,
            mean_degree / num_query))
    if found_cnt:
        print("Among Hits@Inf: Hits@1: {}, Hits@3: {}, Hits@10: {}, MR: {}, MRR: {}, degree: {}".format(
            hit_1 / found_cnt,
            hit_3 / found_cnt,
            hit_10 / found_cnt,
            MR_found / found_cnt,
            MRR_found / found_cnt,
            mean_degree_found / found_cnt))
    else:
        print('No subgraph found the ground truth!!')

    performance_key = ['HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw', 'HITS_INF', 'MRR_raw', 'MR_raw',
                       'HITS_1_found', 'HITS_3_found', 'HITS_10_found', 'MRR_found',
                       'HITS_1_fil', 'HITS_3_fil', 'HITS_10_fil', 'MRR_fil', 'MR_fil',
                       'HITS_1_fil_t', 'HITS_3_fil_t', 'HITS_10_fil_t', 'MRR_fil_t', 'MR_fil_t']
    performance = [hit_1 / num_query, hit_3 / num_query, hit_10 / num_query, found_cnt / num_query, MRR_total / num_query, MR_total/num_query,
                   hit_1 / found_cnt, hit_3 / found_cnt, hit_10 / found_cnt, MRR_found / found_cnt,
                   hit_1_fil / num_query, hit_3_fil / num_query, hit_10_fil / num_query, MRR_total_fil / num_query, MR_total_fil / num_query,
                   hit_1_fil_t / num_query, hit_3_fil_t / num_query, hit_10_fil_t / num_query, MRR_total_fil_t / num_query, MR_total_fil_t / num_query]
    performance_dict = {k: float(v) for k, v in zip(performance_key, performance)}
    checkpoint_dir = os.path.dirname(checkpoint)
    _, epoch = os.path.basename(checkpoint).split("_")
    if args.mongo or args.sqlite:
        dbDriver.test_evaluation(checkpoint_dir, epoch[:-3], performance_dict)
        dbDriver.close()
    import pickle
    with open('tmp_attention_vis_with_time_eval.pickle', 'wb') as handle:
        pickle.dump(model.attention_vis, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved')
    
