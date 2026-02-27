import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import tensorboard
import numpy as np
from tqdm import tqdm
import time
from functools import partial
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

# from args import get_args, set_properties_to_args
from dataset import get_datasets
from model_cuda_hpo import T_RED_GNN
from util import *
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,5,6"
class Options(object):
    pass
# args = Options

def train_one_epoch(model, dataloader, optimizer, data_args, epoch):
    model.train()
    epoch_count = 0
    epoch_loss = 0.
    epoch_correct1, epoch_correct3, epoch_correct10 = 0., 0., 0.

    with tqdm(dataloader, desc=f"Train Ep {epoch}", mininterval=60) as tq:
        for batch in tq:
            batch["head"] = batch["head"].to(data_args.device)
            batch["relation"] = batch["relation"].to(data_args.device)
            batch["tail"] = batch["tail"].to(data_args.device)
            batch["time"] = batch["time"].to(data_args.device)
            # batch["graph"].to(args.device)
            
            score = model(batch)
            predicted_prob = F.softmax(score, dim=1)
            # predicted_prob = attention_history[-1].transpose(0, 1)

            # Compute loss
            loss = F.nll_loss(torch.log(predicted_prob + 1e-12), batch["tail"])
            # pos_scores = predicted_prob[[torch.arange(len(predicted_prob), device=args.device), batch["tail"]]]
            # max_n = torch.max(predicted_prob, 1, keepdim=True)[0]
            # loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(predicted_prob - max_n),1)))


            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss = loss.item()

            if loss != loss:
                print(f"Nan came up at epoch {epoch}")
                exit()

            epoch_loss += loss * batch["head"].size(0)
            epoch_count += batch["head"].size(0)
            avg_loss = epoch_loss / epoch_count

            # Compute hits@k
            epoch_correct1 += hits_at_k(predicted_prob, batch["tail"], k=1)
            # epoch_correct3 += hits_at_k(predicted_prob, batch["tail"], k=3)
            # epoch_correct10 += hits_at_k(predicted_prob, batch["tail"], k=10)

            avg_hits1 = epoch_correct1 / epoch_count
            # avg_hits3 = epoch_correct3 / epoch_count
            # avg_hits10 = epoch_correct10 / epoch_count

            # tq.set_postfix({'Avg loss': avg_loss}, refresh=False)
            # writer.add_scalar('Loss/Train_Avg_Loss', avg_loss,
            #                     global_step=global_count + epoch_count)
            # writer.add_scalar('Metric/Train_hits@1', avg_hits1,
            #                     global_step=global_count + epoch_count)
            # writer.add_scalar('Metric/Train_hits@3', avg_hits3,
            #                     global_step=global_count + epoch_count)
            # writer.add_scalar('Metric/Train_hits@10', avg_hits10,
            #                     global_step=global_count + epoch_count)
            # break

    return avg_loss, avg_hits1

def run_model(config, filenames=None):
    data_args = Options
    data_args.test = False
    data_args.ckpt_dir = 'ckpt'
    # args.ckpt = 'ckpt/0.48462.10.tar'
    data_args.ckpt = ''
    # args.tensorboard_dir = f'runs/{int(time.time())}'
    data_args.epoch = 5
    # args.grad_clip = np.inf
    data_args.dataset = '/home/qiu/Code/KG/T-GAP-RED/data/icews14_aug'
    data_args.device = f'cuda:{get_free_gpu()}'
    print(data_args.device)
    # args.batch_size = 20
    # args.lr = 1e-2
    # args.decay_rate = 0.994
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    hps = Options
    hps.batch_size = config['batch_size']
    hps.lr = 10**config['lr']
    hps.decay_rate = config['decay_rate']
    hps.hidden_dim = config['hidden_dim']
    hps.attn_dim = config['attn_dim']
    hps.act = config['act']
    hps.n_layer = config['n_layer']

    # filenames = ['data/icews14_aug/train.txt', 'data/icews14_aug/valid.txt', 'data/icews14_aug/test.txt']
    # filenames = [os.path.abspath(i) for i in filenames]

    # os.makedirs(args.tensorboard_dir, exist_ok=True)
    # os.makedirs(args.ckpt_dir, exist_ok=True)

    # Configure dataset and dataloader
    train_dataset, _, _ = get_datasets(filenames, data_args.device)
    train_dataloader = DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate, num_workers=4, pin_memory=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
    #                               collate_fn=valid_dataset.collate, num_workers=4, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                              collate_fn=valid_dataset.collate, num_workers=4, pin_memory=True)

    data_args.entity_vocab = train_dataset.kg.entity_vocab
    data_args.relation_vocab = train_dataset.kg.relation_vocab
    data_args.time_vocab = train_dataset.kg.time_vocab
    data_args.graph = train_dataset.graph

    # try:
    best_avg_hits1 = -1.
    # summary_writer = tensorboard.SummaryWriter(log_dir=args.tensorboard_dir)
    # summary_writer.add_text("Args", str(args), 0)
    model = T_RED_GNN(data_args, hps)
    # model = nn.DataParallel(model)
    model.to(data_args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hps.decay_rate)

    for epoch in range(1, 6):
        avg_loss, avg_hits1 = train_one_epoch(model, train_dataloader, optimizer, data_args, epoch)

        # valid_loss, valid_hits1, _, _, _ = evaluate(model, valid_dataloader, args, epoch, summary_writer)
        # scheduler.step(valid_loss)
        scheduler.step()

        if best_avg_hits1 < avg_hits1:
            best_avg_hits1 = avg_hits1
    # except RuntimeError:
    #     best_avg_hits1 = 0
        # print('run out of memory')
    #     return {'loss': -best_avg_hits1, 'status': STATUS_OK}
        tune.report(loss=avg_loss, hit1=avg_hits1)
    print('Finish Training!')
    # with open()
    # return {'loss': -best_avg_hits1, 'status': STATUS_OK}

def main(num_samples=10, max_num_epochs=5, gpus_per_trial=1):
    # data_dir = os.path.abspath("./data")
    filenames = ['data/icews14_aug/train.txt', 'data/icews14_aug/valid.txt', 'data/icews14_aug/test.txt']
    filenames = [os.path.abspath(i) for i in filenames]
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "decay_rate": tune.uniform(0.98, 1),
        "hidden_dim": tune.choice([16, 32, 64]),
        "attn_dim": tune.choice([5, 10, 20, 30]),
        "act": tune.choice(['idd', 'relu', 'tanh', 'sigmoid']),
        "n_layer": tune.choice([3, 4, 5])
    }
    scheduler = ASHAScheduler(
        metric="hit1",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "hit1"])
    result = tune.run(
        partial(run_model, filenames=filenames),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("hit1", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final HIT@1: {}".format(
        best_trial.last_result["hit1"]))
    # print("Best trial final validation accuracy: {}".format(
        # best_trial.last_result["accuracy"]))

    # best_trained_model = model(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    print(best_trial)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main()
    # os.environ["OMP_NUM_THREADS"] = "10"
    # os.environ["MKL_NUM_THREADS"] = "10"
    # os.environ["OPENBLAS_NUM_THREADS"] = "10"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    # os.environ["NUMEXPR_NUM_THREADS"] = "10" 
    # torch.set_num_threads(10)

    # space4kg = {
    #     'lr': hp.uniform('lr', -4, -2),
    #     # 'lamb': hp.uniform('lamb', -5, -3),
    #     'decay_rate': hp.uniform('decay_rate', 0.99, 1),
    #     'dropout': hp.uniform('dropout', 0, 0.3),
    #     'hidden_dim': hp.choice('hidden_dim', [16, 32, 64]),
    #     #'init_dim': hp.choice('init_dim', [5, 10, 25, 50]),
    #     'attn_dim': hp.choice('attn_dim', [5]),
    #     'act': hp.choice('act', ['idd', 'relu', 'tanh']),
    #     #'n_layer': hp.choice('n_layer', [1,2,3,4,5,6]),
    #     'n_layer': hp.choice('n_layer', [3,4,5]),
    #     #'n_batch': hp.choice('n_batch', [10,20,50,100])
    #     #'n_batch': hp.choice('n_batch', [5, 10, 20, 30, 50]),
    #     'batch_size': hp.choice('n_batch', [10, 20, 30]),
    #     #'n_samples': hp.choice('n_samples', [2, 5, 10, 15, 20]),
    # }
    # config = {
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([2, 4, 8, 16]),
    #     "decay_rate": tune.uniform(0.98, 1),
    #     "hidden_dim": tune.choice([16, 32, 64]),
    #     "attn_dim": tune.choice([5, 10, 20, 30]),
    #     "act": tune.choice(['idd', 'relu', 'tanh', 'sigmoid']),
    #     "n_layer": tune.choice([3, 4, 5])
    # }


    # trials = Trials()
    # best = fmin(run_model, space4kg, algo=partial(tpe.suggest, n_startup_jobs=20), max_evals=200, trials=trials)
    # print(best)

    # gpus_per_trial = 2
    # hyperopt_search = HyperOptSearch(space4kg, metric="mean_accuracy", mode="max")

    # analysis = tune.run(run_model, config=hyperopt_search, resources_per_trial={'gpu': 1})







