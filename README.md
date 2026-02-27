# RED-GNN

## Instructions

A quick instruction is given for readers to reproduce the whole process.

Requirements 

- pytorch  1.9.1+cu102
- torch_scatter 2.0.9

## Static KG

This part of the code contains the Transductive and Inductive settings of Static KG.

```
cd Static
```

For transductive reasoning

    cd transductive
    python -W ignore train.py --data_path=data/YAGO



For inductive reasoning

    cd inductive
    python -W ignore train.py --data_path=data/fb237_v2



#### Data splition in transductive setting

We follow the rule mining methods, i.e., [Neural-LP](https://github.com/fanyangxyz/Neural-LP) and [DRUM](https://github.com/alisadeghian/DRUM), to randomly split triplets in the original `train.txt` file into two files `facts.txt` and `train.txt` with ratio 3:1. This step is to make sure that the query triplets will not be leaked in the fact triplets used in RED-GNN. Empirically, increasing the ratio of facts, e.g. from 3:1 to 4:1, will lead to better performance.



## **Temporal KG**

This project provides implementations for temporal knowledge graph reasoning tasks under both **interpolation** and **extrapolation** settings. It supports datasets such as ICEWS14, ICEWS05-15, Wikidata11k, and YAGO. The original folder `T-GAP-RED` has been renamed to `interpolation`, and `T-xERTE-RED` has been renamed to `extrapolation`.

### How to Run

#### Interpolation (in `interpolation/`)

* ICEWS14

  ```bash
  python interpolation/main.py
  ```

* ICEWS05-15

  ```bash
  python interpolation/main_icews05-15.py
  ```

* Wikidata11k

  ```bash
  python interpolation/main_wikidata11k.py
  ```

#### Extrapolation (in `extrapolation/`)

* ICEWS14\_forecasting

  ```bash
  python extrapolation/main.py \
    --warm_start_time 48 \
    --dataset ICEWS14_forecasting \
    --epoch 40 \
    --device 0 \
    --batch_size 2
  ```

* YAGO

  ```bash
  python extrapolation/main.py \
    --warm_start_time 48 \
    --dataset YAGO \
    --epoch 40 \
    --device 7 \
    --batch_size 2
  ```
