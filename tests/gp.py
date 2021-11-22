import os
from argparse import Namespace
import numpy as np

from probabilistic_forecast.nn_mc import NN_MC
from probabilistic_forecast.ensemble import Deep_Ensemble
from probabilistic_forecast.swag import SWAG
from probabilistic_forecast.lstm_mc import LSTM_MC
from probabilistic_forecast.bnn import BNN
from probabilistic_forecast.gnn_mc import GNN_MC
from probabilistic_forecast.nn_standard import NN_Standard

from probabilistic_forecast.utils.data_utils import data_loader
from probabilistic_forecast.utils.torch_utils import torch_loader
from probabilistic_forecast.utils.plot_utils import plot_results, rescale

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

import scipy

from sklearn.metrics import make_scorer

#utils
from functools import partial

import numpy as np
import pandas as pd
from scipy import special
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)


import torch
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'grid.linestyle': '--'})


def root_mean_squared_error(y_true , y_pred):
    return np.sqrt(np.power((y_true - y_pred), 2).mean())

def root_mean_squared_error(y_true , y_pred):
    return np.sqrt(np.power((y_true - y_pred), 2).mean())
       
def mean_absolute_error(y_true , y_pred):
    return  np.absolute((y_true - y_pred)).mean()
    
def prediction_interval_coverage_probability(y_true, y_lower, y_upper):
    k_lower= np.maximum(0, np.where((y_true - y_lower) < 0, 0, 1))
    k_upper = np.maximum(0, np.where((y_upper - y_true) < 0, 0, 1)) 
    PICP = np.multiply(k_lower, k_upper).mean()
    return PICP

def mean_prediction_interval_width(y_lower, y_upper):
    return (y_upper - y_lower).mean()


def data_loader(task, sequence_length=24, forecast_horizon=24, start_train='2019-01-01', end_train='2019-12-31', start_test='2020-01-01', end_test='2020-02-01'):
    data_dir="../dataset"

    df_aq = pd.read_csv(data_dir+'/air_quality_measurements.csv',index_col='time',  parse_dates=True)
    df_weather= pd.read_csv(data_dir+'/weather.csv', index_col='time', parse_dates=True)
    df_traffic = pd.read_csv(data_dir+'/traffic.csv', index_col='Time', parse_dates=True)
    df_traffic = df_traffic.add_prefix('traffic_')
    df_street_cleaning = pd.read_csv(data_dir+'/street_cleaning.csv', index_col='time', parse_dates=True)
    df_street_cleaning = df_street_cleaning.add_prefix('street_cleaning_')

    columns = df_aq.columns.values.copy()

    if task == 'time_series_forecast':
        reg_cols = [col for col in columns if ('_class' not in col) and ('_threshold' not in col)]
        y_reg = df_aq[reg_cols].copy()
        # alleviate exponential effects, the target variable is log-transformed as per the Uber paper. add 1 to avi0d log(0)
        # use np.exp(y)-1 to refers.
        y_reg= np.log(y_reg+1)
        y_reg_mean=y_reg.mean()
        y_reg_std=y_reg.std()
        y_reg = (y_reg- y_reg_mean)/y_reg_std
        y = y_reg
        stats = {'y_reg_mean': y_reg_mean.values, 'y_reg_std': y_reg_std.values, 'reg_cols': reg_cols}

    elif task == 'threshold_exceedance_forecast':
        thre_cols = [col for col in columns if ('_threshold' in col)]
        y_thre = df_aq[thre_cols].copy()
        y = y_thre
        stats = {'thre_cols': thre_cols}

    df_weather_mean=df_weather.mean()
    df_weather_std=df_weather.std()
    df_weather = (df_weather - df_weather_mean)/df_weather_std

    df_traffic_mean=df_traffic.mean()
    df_traffic_std=df_traffic.std()
    df_traffic = (df_traffic - df_traffic_mean)/df_traffic_std

    start_train = pd.to_datetime(start_train)
    end_train = pd.to_datetime(end_train)

    y_train = y[start_train:end_train].copy()
    train_index = y_train.index
    df_traffic_train, df_weather_train, df_street_cleaning_train = df_traffic[start_train:end_train].copy(), df_weather[start_train:end_train].copy(), df_street_cleaning[start_train:end_train].copy()


    timediff = forecast_horizon*np.ones(shape=(y_train.shape[0],))
    #deal with the first day
    for i , diff in enumerate(timediff[:forecast_horizon+1]):
        timediff[i] = min(diff, i )

    latest_idx = train_index - pd.to_timedelta(timediff, unit='H')
    latest_measurement_train= y_train.loc[latest_idx].copy()
    latest_measurement_train.index = train_index
    latest_measurement_train= latest_measurement_train.add_prefix('persistence_')
    X_train = pd.concat([latest_measurement_train, df_traffic_train, df_weather_train, df_street_cleaning_train], axis=1)

    y_train = y_train.values
    X_train = X_train.values

    start_test = pd.to_datetime(start_test) - pd.to_timedelta(sequence_length, unit='H')#to allow for sequence_length
    end_test = pd.to_datetime(end_test)
    y_test = y[start_test:end_test].copy()
    test_index = y_test.index
    df_traffic_test, df_weather_test, df_street_cleaning_test = df_traffic[start_test:end_test].copy(), df_weather[start_test:end_test].copy(), df_street_cleaning[start_test:end_test].copy()


    timediff = forecast_horizon*np.ones(shape=(y_test.shape[0],))
    #deal with the first day
    for i , diff in enumerate(timediff[:forecast_horizon+1]):
        timediff[i] = min(diff, i )

    latest_idx = test_index - pd.to_timedelta(timediff, unit='H')
    latest_measurement_test= y_test.loc[latest_idx].copy()
    latest_measurement_test.index = test_index
    latest_measurement_test= latest_measurement_test.add_prefix('persistence_')
    X_test = pd.concat([latest_measurement_test, df_traffic_test, df_weather_test, df_street_cleaning_test], axis=1)
    
    feature_names = list(X_test.columns)

    y_test = y_test.values
    X_test = X_test.values

    stats['start_test'], stats['end_test'], stats['sequence_length'] = start_test, end_test, sequence_length
    
    return X_train, y_train, X_test, y_test, stats, feature_names

def rescale(y, stats, i,  task='time_series_forecast'):

    if task == 'time_series_forecast':
        y = y * stats['y_reg_std'][i] + stats['y_reg_mean'][i]
        y = np.exp(y)-1
    
    return y

NLL_criterion = torch.nn.GaussianNLLLoss(full=True, reduction='mean')
_normcdf = special.ndtr
def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(x * x) / 2.0)
def crps_gaussian(x, mu, sig):
    x = np.asarray(x)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    # standadized x
    sx = (x - mu) / sig
    # some precomputations to speed up the gradient
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    return crps.mean()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == '__main__':
    task = 'time_series_forecast'
    save_name = 'xgboost_reg'
    sequence_length=0
    forecast_horizon=24
    batch_size=128
    linear_input=False
    start_train='2019-01-01'
    end_train='2019-12-31'
    start_test='2020-01-01'
    end_test='2020-02-01'
    X_train, y_train, X_test, y_test, stats, feature_names =data_loader(task=task,
                                                    sequence_length=sequence_length, 
                                                    forecast_horizon=forecast_horizon, 
                                                    start_train=start_train,
                                                    end_train=end_train,
                                                    start_test=start_test,
                                                    end_test=end_test)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    rows = len(stats['reg_cols'])
    fig, axs = plt.subplots(rows, 1, figsize=(20, 3.5*rows))
    test_index = pd.date_range(start=stats['start_test'], end=stats['end_test'], freq='H')[stats['sequence_length']:]
    xlim = [pd.to_datetime(stats['start_test']) - pd.to_timedelta(stats['sequence_length'], unit='H'), pd.to_datetime(stats['end_test'])]
    training_iter =  100
    for i in range(0, rows):
        train_x = torch.tensor(X_train, dtype=torch.float)
        train_y = torch.tensor(y_train[:, i], dtype=torch.float)
        model = ExactGPModel(train_x, train_y, likelihood)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  
        ], lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for it in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Sensor %d, Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i ,it + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ), end='\r')
            optimizer.step()
        
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x =  torch.tensor(X_test, dtype=torch.float)
            observed_pred = likelihood(model(test_x))
        
        with torch.no_grad():
            std95 = observed_pred.variance.sqrt().mul_(1.96)
            upper = observed_pred.mean.add(std95) 
            lower = observed_pred.mean.sub(std95)
            pred_upper = rescale(upper.numpy(), stats, i)
            pred_lower = rescale(lower.numpy(), stats, i)
            y_pred = rescale(observed_pred.mean.numpy(), stats, i)
            y_true = rescale(y_test[:, i], stats, i)
            
            crps = crps_gaussian(y_test[:, i],observed_pred.mean.numpy(), observed_pred.variance.sqrt().numpy())
            nll = NLL_criterion(torch.tensor(y_test[:, i]), observed_pred.mean, observed_pred.variance).item()
            
            rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
            picp = prediction_interval_coverage_probability(y_true=y_true, y_lower=pred_lower, y_upper=pred_upper)
            mpiw = mean_prediction_interval_width(y_lower=pred_lower, y_upper=pred_upper)
            
            axs[i].fill_between(test_index, pred_lower, pred_upper,  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
            axs[i].plot(test_index, y_pred, color = 'r', label='Forecast')
            axs[i].plot(test_index, y_true, color = 'b', label='Observed value')
            axs[i].plot(test_index, pred_upper, color= 'k', linewidth=0.4)
            axs[i].plot(test_index, pred_lower, color= 'k', linewidth=0.4)
            
            axs[i].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f}, CRPS ={4:0.2f}, NLL = {5:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw, crps, nll))
            print("Monitoring Station ({0}): &{1:0.2f}&{2:0.2f}&{3:0.2f}&{4:0.2f}&{5:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw, crps, nll))
            if '_pm25' in stats['reg_cols'][i]:
                axs[i].set_ylim(-1, 30)
                axs[i].set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
            else:
                axs[i].set_ylim(-1, 40)
                axs[i].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
            axs[i].set_xlim(xlim)
            axs[i].legend(loc="upper left")
    fig.tight_layout()
    plots_dir='../plots/GP'
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(plots_dir + 'regression_all_stations.jpg', bbox_inches='tight')