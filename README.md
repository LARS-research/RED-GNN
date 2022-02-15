# RED-GNN
The code for our paper ["Knowledge Graph Reasoning with Relational Digraph", which has been accepted by ](https://arxiv.org/pdf/2108.06040.pdf)WebConf 2022.



## Instructions

A quick instruction is given for readers to reproduce the whole process.



Requirements 

- pytorch  1.9.1+cu102
- torch_scatter 2.0.9



For transductive reasoning

    cd transductive
    python -W ignore train.py --data_path=data/WN18RR



For inductive reasoning

    cd inductive
    python -W ignore train.py --data_path=data/WN18RR_v1




### Transductive results

| Metrics    | Family | UMLS | WN18RR | FB15k-237 | NELL-995 |
| ---------- | ------ | ---- | ------ | --------- | -------- |
| MRR        | .992   | .964 | .533   | .374      | .543     |
| Hit@1 (%)  | 98.8   | 94.6 | 48.5   | 28.3      | 47.6     |
| Hit@10 (%) | 99.7   | 99.0 | 62.4   | 55.8      | 65.1     |


### Inductive results

We use the full set of negative samples in evaluating the inductive results. This is different from the setting of 50 negative samples in [GraIL](https://arxiv.org/pdf/1911.06962.pdf).

| metrics    | WN-V1 | WN-V2 | WN-V3 | WN-V4 | FB-V1 | FB-V2 | FB-V3 | FB-V4 | NL-V1 | NL-V2 | NL-V3 | NL-V4 |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| MRR        | .701  | .690  | .427  | .651  | .369  | .469  | .445  | .442  | .637  | .419  | .436  | .363  |
| Hit@1 (%)  | 65.3  | 63.3  | 36.8  | 60.6  | 30.2  | 38.1  | 35.1  | 34.0  | 52.5  | 31.9  | 34.5  | 25.9  |
| Hit@10 (%) | 79.9  | 78.0  | 52.4  | 72.1  | 48.3  | 62.9  | 60.3  | 62.1  | 86.6  | 60.1  | 59.4  | 55.6  |



