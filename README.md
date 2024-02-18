# Comparison of EEG stress-prediction using constructive vs static networks.

This repository compares the performance of static and constructive neural networks. Results from the [corresponding writeup (V1)](https://openreview.net/forum?id=0SZNu3wgnSF) can be reproduced as follows:
1. Run `data_prep.ipynb` to process `data/features_by_participant.xlsx`.
2. Run `train_test.ipynb`. The training cell must be re-run for each dataset, which is done by changing the variable `dataset` at the top of the cell.

Note that 5-run k-fold cross-validation can take a while to run.

In addition to packages from the standard library, you'll need:
> `sklearn, torch, matplotlib, seaborn, numpy, pandas, ipympl, openpyxl`.

All other python files are helpers:
- `constr_casc.py` defines the constructive cascade network architecture (Casper).
- `static_nets.py` defines the static network architecture (both for logistic regression and MLP training).
- `performance_metrics.py` contains helpers for tracking the performance of each model.
- `data_prep.ipynb` processes the original dataset.

To track results across sessions, performance metrics are saved to `.csv` files under `results/`. The results included with the current repository reflect those presented in the writeup linked above. Each time a model is run/evaluated, **results for the corresponding dataset and model ID will be overwritten in the appropriate CSV file**, unless `recompute_model_perfs` is removed from the end of the training cell.

For references, see the writeup.