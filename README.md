# Probabilistic Deep Forecast

PyTorch implementation of a paper, titled: **Probabilistic Deep Learning to Quantify Uncertainty in Air Quality Forecasting** [![arXiv](https://img.shields.io/badge/CS.ML-arXiv%3A2112.02622-B31B1B)](https://arxiv.org/abs/2112.02622).



## Introduction

In this work, we develop a set of deep probabilistic models for air quality forecasting that quantify both aleatoric and epistemic uncertainties and study how to represent and manipulate their predictive uncertainties. In particular:

    1. We conduct a broad empirical comparison and exploratory assessment of state-of-the-art techniques in deep probabilistic learning applied to air quality forecasting. Through exhaustive experiments, we describe training these models and evaluating their predictive uncertainties using various metrics for regression and classification tasks.
    2. We improve uncertainty estimation using adversarial training to smooth the conditional output distribution locally around training data points.
    3. We apply uncertainty-aware models that exploit the temporal and spatial correlation inherent in air quality data using recurrent and graph neural networks.
    4. We introduce a new state-of-the-art example for air quality forecasting by defining the problem setup and selecting proper input features and models.

| <img src="plots/epistemic_vs_aleatoric.png" alt="drawing" width="800"/> |
|:--:| 
| Decision score as a function of normalized aleatoric and epistemic confidence thresholds . See [animation video here](https://youtu.be/cGLsYwtn6Ng)|

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
│   ├── nn_standard.py (class definition for the standard neural network model without MC dropout)
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
│   ├── weather.csv  (dataset of weather observations)
│   └── visualize_data.py  (script to visualize all dataset)
├── main.py (main function with argument parsing to load data, build a model and evaluate (or train))
├── tests/
│   └── confidence_reliability.py (script to evaluate the reliability of confidence estimates of pretrained models)
│   └── epistemic_vs_aleatoric.py (script to show the impact of quantifying both epistemic and aleatoric uncertainties)
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


|<img src="plots/GNN_MC/regression.jpg" alt="drawing" width="800"/>|
|:--:| 
|PM-value regression using Graph Neural Network with MC dropout|

### Threshold-exceedance prediction

|<img src="plots/BNN/classification.jpg" alt="drawing" width="800"/>|
|:--:| 
|Threshold-exceedance prediction using Bayesian neural network (BNN)|


### Confidence Reliability

To evaluate the confidence reliability of the considered probabilistic models, run the following command:

```bash
python tests/confidence_reliability.py
```

It will generate the following plots:


|<img src="plots/regression_confidence_reliability_all_stations.jpg" alt="drawing" width="800"/>|
|:--:| 
|Confidence reliability of probabilistic models in PM-value regression task in all monitoring stations.|



|<img src="plots/classification_confidence_reliability_all_stations.jpg" alt="drawing" width="800"/>|
|:--:| 
|Confidence reliability of probabilistic models in threshold-exceedance prediction task in all monitoring stations.|

### Epistemic and aleatoric uncertainties in decision making

To evaluate the impact of quantifying both epistemic and aleatoric uncertainties in decision making, run the following command:

```bash
python tests/epistemic_vs_aleatoric.py
```
It will generate the following plots:

Decision score in a non-probabilistic model <br /> as a function of only aleatoric confidence.             |  Decision score in a probabilistic model as a function <br />  of both epistemic and aleatoric confidences. 
:-------------------------:|:-------------------------:
<img src="plots/decision_making_standard.jpg" alt="drawing" width="400"/> |  <img src="plots/decision_making_probabilistic.jpg" alt="drawing" width="400"/>

It will also generate an `.vtp` file, which can be used to generate a 3D plot with detailed rendering and lighting in [ParaView](https://www.paraview.org/).





## Training Models

Train a single model, for example:
```bash
python main.py --model=SWAG --task=regression --mode=train --n_epochs=3000 --adversarial_training
```
or train all models:
```bash
bash train_all_models.sh
```

|<img src="pretrained/BNN/train_bnn_reg.jpg" alt="drawing" width="800"/>|
|:--:| 
|Learning curve of training a BNNs model to forecast PM-values. **Left:** negative log-likelihood loss,<br />  **Center:** KL loss estimated using MC sampling, **Right:** learning rate of exponential decay.|

## Dataset

Run the following command to visualize all data

```bash
python dataset/visualize_data.py
```

It will generate plots in the "dataset folder". For example:

| <img src="/plots/dataset_visualization/aq_index.jpg" alt="drawing" width="800"/> |
|:--:| 
|Air quality level over two years in one representative monitoring station (Elgeseter) in Trondheim, Norway|




## Attribution

* Parts of the SWAG code is based on the official code for the paper "[A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/2103.07719)": [https://github.com/wjmaddox/swa_gaussian](https://github.com/wjmaddox/swa_gaussian).

* Parts of the GNN code is based on the official code for the paper "[Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://arxiv.org/abs/1902.02476)": [https://github.com/microsoft/StemGNN](https://github.com/microsoft/StemGNN).

* Parts of the BNN code is based on [https://github.com/JavierAntoran/Bayesian-Neural-Networks](https://github.com/JavierAntoran/Bayesian-Neural-Networks).

* The function `h5_to_vt` in `epistemic_vs_aleatoric.py` that convert h5 file to vtp files to be used by ParaView, is based on [https://github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape).

* The air quality dataset is a part of the open database of air quality measurements offered by the Norwegian Institute for Air Research (NILU) [https://www.nilu.com/open-data/](https://www.nilu.com/open-data/).

* The meteorological data are based on historical weather and climate data offered by the Norwegian Meteorological Institute [https://frost.met.no](https://frost.met.no).

* The traffic data is based on aggregated traffic volumes offered by the Norwegian Public Roads Administration [https://www.vegvesen.no/trafikkdata/start/om-api](https://www.vegvesen.no/trafikkdata/start/om-api).
