import os
import sys
import math
import pprint
import time

import ray
import wandb
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
from ray.tune.logger import DEFAULT_LOGGERS

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class Config:
    pass

from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
webhook = "https://oapi.dingtalk.com/robot/send?access_token=c406b57a6222df65db4386734146c32914d05ae564487fc375c0f4dc77b3e4df"
secret = "SEC59ed463ef83a276b19d78165e6037f6d07be6eea4a936e2e75f3690fc2243038"

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

def test(cfg, solver):
    # solver.model.split = "valid"
    # solver.evaluate("valid")
    solver.model.split = "test"
    return solver.evaluate("test")

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
    return events

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

def objective(opt):
    wandb = setup_wandb(opt, project="T-RED-GNN")
    sys.path.append('/home/qiuhaiquan/Code/KG/T-xERTE-RED')
    from model_cuda import T_RED_GNN
    from utils import Data, save_config, get_git_version_short_hash, get_git_description_last_commit, load_checkpoint, new_checkpoint, get_time_offset_list
    import torch
    import torch.nn.functional as F
    import numpy as np
    from segment import segment_rank_fil
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    gpu_id = ray.get_gpu_ids()
    print(gpu_id)
    device = 'cuda'
    print('========================= device is ')
    print(device)

    dataset = "ICEWS0515_forecasting"

    cfg = Config()
    cfg.n_layer = opt['n_layer']
    cfg.hidden_dim = opt['hidden_dim']
    cfg.attn_dim = opt['attn_dim']
    cfg.act = opt['act']
    cfg.dropout = opt['dropout']
    cfg.device = device
    cfg.lr =10**opt['lr']
    cfg.batch_size = opt['batch_size']
    cfg.patience = 3

    contents = Data(dataset=dataset, add_reverse_relation=True)
    cfg.contents = contents
    if 'yago' in dataset.lower():
        time_granularity = 1
    elif 'icews' in dataset.lower():
        time_granularity = 24
    else:
        raise ValueError
    time_offset_list = get_time_offset_list(contents.data)
    cfg.time_offset_list = time_offset_list

    model = T_RED_GNN(cfg)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1*1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=cfg.patience)
    start_epoch = 0

    sp2o = contents.get_sp2o()
    val_spt2o = contents.get_spt2o('valid')
    train_inputs = prepare_inputs(contents, start_time=48, tc=None)
    val_inputs = prepare_inputs(contents, dataset='valid', tc=None)
    analysis_data_loader = DataLoader(val_inputs, batch_size=cfg.batch_size, collate_fn=collate_wrapper,
                                 pin_memory=False, shuffle=True)
    analysis_batch = next(iter(analysis_data_loader))
    best_epoch = 0
    best_val = 0
    total_iter = 0
    for epoch in range(0, 2):
        training_time_epoch_start = time.time()
        # load data
        train_inputs = prepare_inputs(contents, start_time=48, tc=None)
        train_data_loader = DataLoader(train_inputs, batch_size=cfg.batch_size, collate_fn=collate_wrapper,
                                       pin_memory=False, shuffle=True)

        running_loss = 0.
        running_hit1 = 0.
        for batch_ndx, sample in enumerate(train_data_loader):
            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            score, (entity_att_score, entities) = model(sample)
            target_idx_l = torch.from_numpy(sample.target_idx).type(torch.LongTensor).to(device)

            predicted_prob = F.softmax(score, dim=1)
            # one_hot_label = torch.from_numpy(
            # np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(params.device)
            # loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
            loss = F.nll_loss(torch.log(predicted_prob + 1e-12), target_idx_l)
            # loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size, args.gradient_iters_per_update, args.loss_fn)

            predict_lablel = torch.argmax(score, axis=1)
            avg_hit1 = torch.mean((predict_lablel == target_idx_l).float()).item()
            running_hit1 = running_hit1 + avg_hit1

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            model.zero_grad()

            running_loss += loss.item()
            total_iter = total_iter + 1
            # print(str_time_cost(time_cost))
        running_loss /= batch_ndx + 1

        print("training time per epoch without validation: " + str(time.time() - training_time_epoch_start))
        model.eval()

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

            val_data_loader = DataLoader(val_inputs, batch_size=cfg.batch_size, collate_fn=collate_wrapper,
                                         pin_memory=False, shuffle=True)

            val_running_loss = 0
            val_running_hit1_sum = 0
            val_loss_list = []
            valid_cnt = 0
            for batch_ndx, sample in enumerate(val_data_loader):
                model.eval()

                src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
                num_query += len(src_idx_l)
                # degree_batch = model.ngh_finder.get_temporal_degree(src_idx_l, cut_time_l)
                # mean_degree += sum(degree_batch)

                score, (entity_att_score, entities) = model(sample)
                target_idx_l = torch.from_numpy(sample.target_idx).type(torch.LongTensor).to(device)

                predicted_prob = F.softmax(score, dim=1)

                # one_hot_label = torch.from_numpy(
                #     np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(params.device)
                # loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
                loss = F.nll_loss(torch.log(predicted_prob + 1e-12), target_idx_l)
                # loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size,
                #                   args.gradient_iters_per_update, args.loss_fn)
                valid_cnt += 1

                predict_lablel = torch.argmax(score, axis=1)
                avg_hit1 = torch.mean((predict_lablel == target_idx_l).float()).item()
                val_running_hit1_sum += avg_hit1

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
            print(
                "Filtered performance (time dependent): Hits@1: {}, Hits@3: {}, Hits@10: {}, MRR: {}".format(
                    hit_1_fil_t / num_query,
                    hit_3_fil_t / num_query,
                    hit_10_fil_t / num_query,
                    MRR_total_fil_t / num_query))

            scheduler.step(val_running_loss)
        session.report({"mrr": MRR_total_fil_t / num_query, "h1": hit_1_fil_t / num_query, "h3":hit_3_fil_t / num_query, "h10": hit_10_fil_t / num_query})
        wandb.log({"mrr": MRR_total_fil_t / num_query, "h1": hit_1_fil_t / num_query, "h3":hit_3_fil_t / num_query, "h10": hit_10_fil_t / num_query})
        

space = {
    "lr": tune.uniform(-4, -1 ),
    'n_layer': tune.choice([3, 5, 7]),
    'hidden_dim': tune.choice([32, 64, 128]),
    'attn_dim': tune.choice([5, 10]),
    'act': tune.choice(['relu', 'tanh', 'idd', 'sigmoid']),
    'dropout': tune.uniform(0, 0.3),
    'batch_size': tune.choice([10, 30, 50]),
    "wandb": {
            "project": "T-RED-GNN",
            "log_config": True,
            "entity": "qqpp",
    }
}

if __name__ == '__main__':

    ray.init(num_gpus=3)

    # sched = AsyncHyperBandScheduler(
    #     time_attr="training_iteration", max_t=200, grace_period=20
    # )

    hyperopt_search = HyperOptSearch(metric='mrr', mode='max', n_initial_points=50)


    tuner = tune.Tuner(
        tune.with_resources(
            objective, 
            resources={'cpu': 10, 'gpu': 0.5}
            ), 
        param_space=space, 
        tune_config=tune.TuneConfig(
            num_samples=50,
            search_alg=hyperopt_search,            
        ),
        run_config=air.RunConfig(
            callbacks=[WandbLoggerCallback(project="T-RED-GNN")])
        )
    results = tuner.fit()
    result_df = results.get_dataframe()
    result_df.to_csv('result.csv', index=False, sep='\t')
    print(result_df['mrr'].max())
    print(results.get_best_result(metric="mrr", mode="max"))
    print(results.get_best_result(metric="mrr", mode="max").config)


    xiaoding = DingtalkChatbot(webhook, secret=secret)
    xiaoding.send_text(msg='THU_4P finished HPO of T-RED-GNN', is_at_all=True)


# Result(metrics={'mrr': 0.9593558311462402, 'h1': 0.9593558311462402, 'h10': 0.9938650131225586, 't_mrr': 0.9675185084342957, 't_h1': 0.9485628008842468, 't_h10': 0.9947050213813782, 'done': True, 'trial_id': 'a3d398dc', 'experiment_tag': '48_activation=relu,batch_size=10,degree=200,dependent=True,lr=-2.7135,msg_func=distmult,short_cut=False'}, error=None, log_dir=PosixPath('/home/qiuhaiquan/ray_results/objective_2023-03-18_00-07-34/objective_a3d398dc_48_activation=relu,batch_size=10,degree=200,dependent=True,lr=-2.7135,msg_func=distmult,short_cut=False_2023-03-18_00-37-03'))