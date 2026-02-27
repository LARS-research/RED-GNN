import os
# the mode argument of the os.makedirs function may be ignored on some systems
# umask (user file-creation mode mask) specify the default denial value of variable mode,
# which means if this value is passed to makedirs function,
# it will be ignored and a folder/file with d_________ will be created
# we can either set the umask or specify mode in makedirs

# oldmask = os.umask(0o770)
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import setproctitle

setproctitle.setproctitle('TKG@QiuHaiquan')
PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder, Measure, save_config, get_git_version_short_hash, get_git_description_last_commit, load_checkpoint, new_checkpoint, get_time_offset_list
from model_cuda import T_RED_GNN
from segment import *
from database_op import DBDriver

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
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'valid':
        contents_dataset = contents.valid_data
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'test':
        contents_dataset = contents.test_data
        assert start_time < max(contents_dataset[:, 3])
    else:
        raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")
    events = np.vstack([np.array(event) for event in contents_dataset if event[3] >= start_time])
    if args.timer:
        tc['data']['load_data'] += time.time() - t_start
    return events


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.src_idx = np.array(transposed_data[0], dtype=np.int32)
        self.rel_idx = np.array(transposed_data[1], dtype=np.int32)
        self.target_idx = np.array(transposed_data[2], dtype=np.int32)
        self.ts = np.array(transposed_data[3], dtype=np.int32)
        self.event_idx = np.array(transposed_data[-1], dtype=np.int32)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_idx = self.src_idx.pin_memory()
        self.rel_idx = self.rel_idx.pin_memory()
        self.target_idx = self.target_idx.pin_memory()
        self.ts = self.ts.pin_memory()
        self.event_idx = self.event_idx.pin_memory()

        return self

    def __str__(self):
        return "Batch Information:\nsrc_idx: {}\nrel_idx: {}\ntarget_idx: {}\nts: {}".format(self.src_idx, self.rel_idx, self.target_idx, self.ts)


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


parser = argparse.ArgumentParser()
parser.add_argument('--emb_dim', type=int, default=[256, 128, 64, 32], nargs='+', help='dimension of embedding for node, realtion and time')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--DP_steps', type=int, default=3, help='number of DP steps')
parser.add_argument('--DP_num_edges', type=int, default=15, help='number of edges at each sampling')
parser.add_argument('--max_attended_edges', type=int, default=40, help='max number of edges after pruning')
parser.add_argument('--ratio_update', type=float, default=0, help='ratio_update: when update node representation: '
                                                                  'ratio * self representation + (1 - ratio) * neighbors, '
                                                                  'if ratio==0, GCN style, ratio==1, no node representation update')
parser.add_argument('--dataset', type=str, default=None, help='specify data set')
parser.add_argument('--whole_or_seen', type=str, default='whole', choices=['whole', 'seen', 'unseen'], help='test on the whole set or only on seen entities.')
parser.add_argument('--warm_start_time', type=int, default=48, help="training data start from what timestamp")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--device', type=int, default=2, help='-1: cpu, >=0, cuda device')
parser.add_argument('--sampling', type=int, default=3,
                    help='strategy to sample neighbors, 0: uniform, 1: first num_neighbors, 2: last num_neighbors, 3: time-difference weighted')
parser.add_argument('--load_checkpoint', type=str, default=None, help='train from checkpoints')
parser.add_argument('--weight_factor', type=float, default=2, help='sampling 3, scale the time unit')
parser.add_argument('--node_score_aggregation', type=str, default='sum', choices=['sum', 'mean', 'max'])
parser.add_argument('--ent_score_aggregation', type=str, default='sum', choices=['sum', 'mean'])
parser.add_argument('--emb_static_ratio', type=float, default=1, help='ratio of static embedding to time(temporal) embeddings')
parser.add_argument('--add_reverse', action='store_true', default=True, help='add reverse relation into data set')
parser.add_argument('--loss_fn', type=str, default='BCE', choices=['BCE', 'CE'])
parser.add_argument('--no_time_embedding', action='store_true', default=False, help='set to stop use time embedding')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--sqlite', action='store_true', default=None, help='save information to sqlite')
parser.add_argument('--mongo', action='store_true', default=None, help='save information to mongoDB')
parser.add_argument('--use_database', action = 'store_true', default=None, help='use database to store experimental')
parser.add_argument('--gradient_iters_per_update', type=int, default=1, help='gradient accumulation, update parameters every N iterations, default 1. set when GPU memo is small')
parser.add_argument('--timer', action='store_true', default=None, help='set to profile time consumption for some func')
parser.add_argument('--debug', action='store_true', default=None, help='in debug mode, checkpoint will not be saved')
parser.add_argument('--diac_embed', action='store_true', help='use entity-specific frequency and phase of time embeddings')
args = parser.parse_args()

params = Options
params.n_layer = 3
params.hidden_dim = 20
params.attn_dim = 30
params.act = 'sigmoid'
params.dropout = 0.0
params.device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
params.patience = 3
if not args.debug:
    import local_config
    save_dir = local_config.save_dir
else:
    save_dir = ''

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(args)

    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    # profile time consumption
    time_cost = None
    if args.timer:
        time_cost = reset_time_cost()

    # init model and checkpoint folder
    start_time = time.time()
    struct_time = time.gmtime(start_time)
    epoch_command = args.epoch

    if args.load_checkpoint is None:
        checkpoint_dir, CHECKPOINT_PATH = new_checkpoint(save_dir, struct_time)
        contents = Data(dataset=args.dataset, add_reverse_relation=args.add_reverse)

        params.contents = contents

        adj = contents.get_adj_dict()
        max_time = max(contents.data[:, 3])

        # construct NeighborFinder
        if 'yago' in args.dataset.lower():
            time_granularity = 1
        elif 'icews' in args.dataset.lower():
            time_granularity = 24
        else:
            raise ValueError
        time_offset_list = get_time_offset_list(contents.data)
        params.time_offset_list = time_offset_list
        # nf = NeighborFinder(adj, sampling=args.sampling, max_time=max_time, num_entities=contents.num_entities,
        #                     weight_factor=args.weight_factor, time_granularity=time_granularity)
        # construct model
        model = T_RED_GNN(params)
        model.to(params.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=params.patience)
        start_epoch = 0
        if not args.debug:
            print("Save checkpoints under {}".format(CHECKPOINT_PATH))
    else:
        checkpoint_dir = os.path.dirname(args.load_checkpoint)
        CHECKPOINT_PATH = os.path.join(save_dir, 'Checkpoints', os.path.dirname(args.load_checkpoint))
        model, optimizer, start_epoch, contents, args = load_checkpoint(
            os.path.join(save_dir, 'Checkpoints', args.load_checkpoint), device=device)
        args.epoch = epoch_command
        start_epoch += 1
        print("Load checkpoints {}".format(CHECKPOINT_PATH))

    # save configuration to database and file system
    if not args.debug and args.use_database:
        dbDriver = DBDriver(useMongo=args.mongo, useSqlite=args.sqlite, MongoServerIP=local_config.MongoServer, sqlite_dir=os.path.join(save_dir, 'tKGR.db'))
        git_hash = get_git_version_short_hash()
        git_comment = get_git_description_last_commit()
        dbDriver.log_task(args, checkpoint_dir, git_hash=git_hash, git_comment=git_comment, device=local_config.AWS_device)
        save_config(args, CHECKPOINT_PATH)

    sp2o = contents.get_sp2o()
    val_spt2o = contents.get_spt2o('valid')
    train_inputs = prepare_inputs(contents, start_time=args.warm_start_time, tc=time_cost)
    val_inputs = prepare_inputs(contents, dataset='valid', tc=time_cost)
    analysis_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                 pin_memory=False, shuffle=True)
    analysis_batch = next(iter(analysis_data_loader))

    writer = SummaryWriter()
    best_epoch = 0
    best_val = 0
    total_iter = 0
    for epoch in range(start_epoch, args.epoch):
        print("epoch: ", epoch)
        training_time_epoch_start = time.time()
        # load data
        train_inputs = prepare_inputs(contents, start_time=args.warm_start_time, tc=time_cost)
        train_data_loader = DataLoader(train_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                       pin_memory=False, shuffle=True)

        running_loss = 0.
        running_hit1 = 0.
        
        for batch_ndx, sample in enumerate(tqdm(train_data_loader)):
            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            score, (entity_att_score, entities) = model(sample)
            target_idx_l = torch.from_numpy(sample.target_idx).type(torch.LongTensor).to(params.device)

            predicted_prob = F.softmax(score, dim=1)
            loss = F.nll_loss(torch.log(predicted_prob + 1e-12), target_idx_l)
            # loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size, args.gradient_iters_per_update, args.loss_fn)

            predict_lablel = torch.argmax(score, axis=1)
            avg_hit1 = torch.mean((predict_lablel == target_idx_l).float()).item()
            running_hit1 = running_hit1 + avg_hit1

            if args.timer:
                t_start = time.time()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if args.timer:
                time_cost['grad']['comp'] += time.time() - t_start

            if args.timer:
                t_start = time.time()
            if (batch_ndx+1) % args.gradient_iters_per_update == 0:
                optimizer.step()
                model.zero_grad()
            if args.timer:
                time_cost['grad']['apply'] += time.time() - t_start

            running_loss += loss.item()
            writer.add_scalar('Loss/Train', running_loss / (batch_ndx+1), total_iter)
            writer.add_scalar('Hit1/Train', running_hit1 / (batch_ndx+1), total_iter)
            total_iter = total_iter + 1
            # print(str_time_cost(time_cost))
            if args.timer:
                time_cost = reset_time_cost()
            break
        running_loss /= batch_ndx + 1

        print("training time per epoch without validation: " + str(time.time() - training_time_epoch_start))
        model.eval()
        if not args.debug:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': args
            }, os.path.join(CHECKPOINT_PATH, 'checkpoint_{}.pt'.format(epoch)))

        if epoch % 1 == 0:
            hit_1 = hit_3 = hit_10 = 0
            hit_1_fil = hit_3_fil = hit_10_fil = 0
            hit_1_fil_t = hit_3_fil_t = hit_10_fil_t = 0
            found_cnt = 0
            MR_total = 0
            MR_found = 0
            MRR_total = 0
            MRR_found = 0
            MRR_total_fil = 0
            MRR_total_fil_t = 0
            num_query = 0
            mean_degree = 0
            mean_degree_found = 0

            val_data_loader = DataLoader(val_inputs, batch_size=5*args.batch_size, collate_fn=collate_wrapper,
                                         pin_memory=False, shuffle=True)

            val_running_loss = 0
            val_running_hit1_list = []
            val_loss_list = []
            for batch_ndx, sample in enumerate(tqdm(val_data_loader)):
                model.eval()

                src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
                num_query += len(src_idx_l)
                # degree_batch = model.ngh_finder.get_temporal_degree(src_idx_l, cut_time_l)
                # mean_degree += sum(degree_batch)

                score, (entity_att_score, entities) = model(sample)
                target_idx_l = torch.from_numpy(sample.target_idx).type(torch.LongTensor).to(params.device)

                predicted_prob = F.softmax(score, dim=1)
                loss = F.nll_loss(torch.log(predicted_prob + 1e-12), target_idx_l)
                # loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size,
                #                   args.gradient_iters_per_update, args.loss_fn)

                predict_lablel = torch.argmax(score, axis=1)
                avg_hit1 = torch.mean((predict_lablel == target_idx_l).float()).item()
                val_running_hit1_list.append(avg_hit1)

                val_running_loss += loss.item()
                val_loss_list.append(loss.item())

                target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = segment_rank_fil(entity_att_score,
                                                                                                     entities,
                                                                                                     target_idx_l.cpu().numpy(),
                                                                                                     sp2o,
                                                                                                     val_spt2o,
                                                                                                     src_idx_l,
                                                                                                     rel_idx_l,
                                                                                                     cut_time_l)
                # mean_degree_found += sum(degree_batch[found_mask])
                hit_1 += np.sum(target_rank_l == 1)
                hit_3 += np.sum(target_rank_l <= 3)
                hit_10 += np.sum(target_rank_l <= 10)
                hit_1_fil += np.sum(target_rank_fil_l <= 1) # unique entity with largest node score
                hit_3_fil += np.sum(target_rank_fil_l <= 3)
                hit_10_fil += np.sum(target_rank_fil_l <= 10)
                hit_1_fil_t += np.sum(target_rank_fil_t_l <= 1) # unique entity with largest node score
                hit_3_fil_t += np.sum(target_rank_fil_t_l <= 3)
                hit_10_fil_t += np.sum(target_rank_fil_t_l <= 10)
                found_cnt += np.sum(found_mask)
                MR_total += np.sum(target_rank_l)
                MR_found += len(found_mask) and np.sum(
                    target_rank_l[found_mask])  # if no subgraph contains ground truch, MR_found = 0 for this batch
                MRR_total += np.sum(1 / target_rank_l)
                MRR_found += len(found_mask) and np.sum(
                    1 / target_rank_l[found_mask])  # if no subgraph contains ground truth, MRR_found = 0 for this batch
                MRR_total_fil += np.sum(1 / target_rank_fil_l)
                MRR_total_fil_t += np.sum(1 / target_rank_fil_t_l)
                # break
            writer.add_scalar('Loss/Valid', np.mean(val_loss_list), epoch)
            writer.add_scalar('Hit1/Valid', np.mean(val_running_hit1_list), epoch)
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
                    mean_degree_found / found_cnt
                    ))
            else:
                print('No subgraph found the ground truth!!')

            performance_key = ['training_loss', 'validation_loss', 'HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw',
                               'HITS_INF', 'MRR_raw', 'HITS_1_found', 'HITS_3_found', 'HITS_10_found', 'MRR_found']
            performance = [running_loss, val_running_loss / (batch_ndx + 1), hit_1 / num_query,
                           hit_3 / num_query,
                           hit_10 / num_query, found_cnt / num_query, MRR_total / num_query, hit_1_fil_t / num_query,
                           hit_3_fil_t / num_query, hit_10_fil_t / num_query, MRR_total_fil_t / num_query]
            performance_dict = {k: float(v) for k, v in zip(performance_key, performance)}

            if not args.debug and args.use_database:
                dbDriver.log_evaluation(checkpoint_dir, epoch, performance_dict)
            if performance[-1] > best_val:
                best_val = performance[-1]
                best_epoch = epoch

            scheduler.step(val_running_loss)

        print("training time per epoch with validation: " + str(time.time() - training_time_epoch_start))

    if not args.debug and args.use_database:
        dbDriver.close()
    print("finished Training")
    print("start evaluation on test set")
    # os.system("python eval.py --load_checkpoint {}/checkpoint_{}.pt --whole_or_seen {} --device {} --mongo".format(checkpoint_dir,
    #                                                 best_epoch, args.whole_or_seen, args.device))
