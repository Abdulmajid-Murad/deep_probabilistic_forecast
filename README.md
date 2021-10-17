# Deep Probabilistic Forecast
PyTorch implementation of deep probabilistic forecast applied to air quality.


## Introduction

In this repo, we build deep probabilistic models that forecast air quality values and predict threshold exceedance events.


## Installation

install probabilistic_forecast' locally  in “editable” mode ( any changes to the original package would reflect directly in your environment, os you don't have to re-insall the package every time you make some changes): 

 ```bash
 pip install -e .
 ```
Use the configuration file  `equirements.txt` to the install the required packages to run this project.

## File Structure

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

* Parts of the GNN code is based ofn the official code for the paper "[Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://arxiv.org/abs/1902.02476)": [https://github.com/microsoft/StemGNN](https://github.com/microsoft/StemGNN)


