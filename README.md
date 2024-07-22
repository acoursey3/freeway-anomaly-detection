# Automatic Freeway Anomalous Event Detection

This repository contains code for lane-level anomaly detection of I-24 traffic data.

## Installation
From the `/code/` directory, run 
`conda env create -f environment.yml`.

Alternatively, it may be preferred to only install the relevant packages as necessary for the parts of the code you want to run. 

## Data Setup

Before running this code, ensure you have the [Freeway Traffic Anomalous Event Detection Dataset](https://github.com/acoursey3/freeway-anomaly-data/) downloaded and in the `/data/` directory. The `nashville_freeway_anomaly.csv` file is all you need.

## Code Index

- [General Graph Creation](./code/DataToGraph.ipynb)
- **Model Training**:
    - [STG-RGCN](./code/TrainSTG-RGCN.ipynb)
    - [STG-GAT](./code/TrainSTG-GAT.ipynb)
    - [GCN-LSTM](./code/TrainGCN-LSTM.ipynb)
    - [GCN](./code/TrainGCN.ipynb)
    - [Transformer](./code/TrainTransformer.ipynb)
    - [MLP](./code/TrainMLP.ipynb)
- **Helper Files**:
    - [Data Utilities](./code/datautils.py)
    - [Metric Computation](./code/metrics.py)
    - [Torch Model Definitions](./code/models.py)
    - [Parameter Classes](./code/parameters.py)
    - [Model Training Helpers](./code/training.py)
- [Hyperparameter Optimization](./code/opt_hyperparams.py)
- [Ablation on Impact of Manual Crash Labels](./code/Manual%20Label%20Ablation.ipynb)
- [Results Visualization](./code/VisualizeResults.ipynb)

## Citation

If you find this code useful for your research, please consider using the following citation.

```
@misc{coursey2024ftaed,
  title={FT-AED: Benchmark Dataset for Early Freeway Traffic Anomalous Event Detection},
  author={Austin Coursey and Junyi Ji and Marcos Quinones-Grueiro and William Barbour and Yuhang Zhang and Tyler Derr and Gautam Biswas and Daniel B. Work},
  year={2024},
  eprint={2406.15283},
  archivePrefix={arXiv},
}
```