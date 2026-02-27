# Temporal Knowledge Graph Reasoning

This project provides implementations for temporal knowledge graph reasoning tasks under both **interpolation** and **extrapolation** settings. It supports datasets such as ICEWS14, ICEWS05-15, Wikidata11k, and YAGO. The original folder `T-GAP-RED` has been renamed to `interpolation`, and `T-xERTE-RED` has been renamed to `extrapolation`.

## How to Run

### Interpolation (in `interpolation/`)

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

### Extrapolation (in `extrapolation/`)

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
