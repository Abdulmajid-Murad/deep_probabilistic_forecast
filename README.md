# Deep Probabilistic Forecast
PyTorch implementation of deep probabilistic forecast applied to air quality.

<img src="plots/epistemic_vs_aleatoric.jpeg" alt="drawing" width="400"/>

[Animation](https://youtu.be/TqjIherIjc4)

## Introduction

In this repo, we build deep probabilistic models that forecast air quality values and predict threshold exceedance events.


## Installation

install probabilistic_forecast' locally  in “editable” mode ( any changes to the original package would reflect directly in your environment, os you don't have to re-insall the package every time you make some changes): 

 ```bash
 pip install -e .
 ```
Use the configuration file  `equirements.txt` to the install the required packages to run this project.

## File Structure


```bash
.
├── probabilistic_forecast/
│   ├── bnn.py (class definition for the Bayesian neural networks model)
│   ├── ensemble.py (class definition for the deep ensemble model)
│   ├── gnn_mc.py (class definition for the graph neural network model with MC dropout)
│   ├── lstm_mc.py (class definition for the LSTM model with MC dropout)
│   ├── nn_mc.py (class definition for the standard neural network model with MC droput)
│   ├── swag.py (class definition for the SWAG model)
│   └── utils/
│       ├── data_utils.py (utility functions for data loading and pre-processing)
│       ├── gnn_utils.py (utility functions for GNN)
│       ├── plot_utils.py (utility functions for plotting training and evaluation results)
│       ├── swag_utils.py  (utility functions for SWAG)
│       └── torch_utils.py (utility functions for torch dataloader, checking if CUDA is available)
├── dataset/
│   ├── air_quality_measurements.csv (dataset of air quality measurements)
│   ├── street_cleaning.csv  (dataset of air street cleaning records)
│   ├── traffic.csv (dataset of traffic volumes)
│   └── weather.csv  (dataset of weather observations)
├── main.py (main function with argument parsing to load data, build a model and evaluate (or train))
├── tests/
│   └── confidence_reliability.py (script to evaluate the reliability of confidence estimates of pretrained models)
├── plots/ (foler containing all evaluation plots)
├── pretrained/ (foler containing pretrained models and training curves plots)
├── evaluate_all_models.sh (bash script for evaluating all models at once)
└── train_all_models.sh (bash script for training all models at once)

```




## Evaluating Pretrained Models

Evaluate a pretrained model, for example:

```bash
python main.py --model=SWAG --task=regression --mode=evaluate  --adversarial_training
```
or evaluate all models:
```bash
bash evaluate_all_models.sh
```
### PM-value regression

![](plots/SWAG/regression_adv.jpg)
![](plots/SWAG/regression.jpg)

### Threshold-exceedance prediction

![](plots/BNN/classification.jpg)

### Confidence Reliability
```bash
python tests/confidence_reliability.py
```

#### PM-value regression
![](plots/regression_confidence_reliability_all_stations.jpg)

#### Threshold-exceedance prediction
![](plots/classification_confidence_reliability_all_stations.jpg)

## Training Models

Train a single model, for example:
```bash
python main.py --model=SWAG --task=regression --mode=train --n_epochs=3000 --adversarial_training
```
or train all models:
```bash
bash train_all_models.sh
```
![](pretrained/BNN/train_bnn_reg.jpg)

## Attribution

* Parts of the SWAG code is based on the official code for the paper "[A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/2103.07719)": [https://github.com/wjmaddox/swa_gaussian](https://github.com/wjmaddox/swa_gaussian)

* Parts of the GNN code is based on the official code for the paper "[Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://arxiv.org/abs/1902.02476)": [https://github.com/microsoft/StemGNN](https://github.com/microsoft/StemGNN)

* Parts of the BNN code is based on [https://github.com/JavierAntoran/Bayesian-Neural-Networks](https://github.com/JavierAntoran/Bayesian-Neural-Networks)


