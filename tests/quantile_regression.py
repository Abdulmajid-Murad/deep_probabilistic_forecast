import xgboost as xgb
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor



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

from matplotlib import cm
cmap = cm.get_cmap('rainbow')

class XGBQuantile(XGBRegressor):
    def __init__(self,quant_alpha=0.95,quant_delta = 1.0,quant_thres=1.0,quant_var =1.0,base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,max_depth=3, min_child_weight=1, missing=1, n_estimators=100,
                    n_jobs=1,  objective='reg:linear', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var
        
        super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
        colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, max_delta_step=max_delta_step,
        max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators,
        n_jobs= n_jobs,  objective=objective, random_state=random_state,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, subsample=subsample)
        
        self.test = None
  
    def fit(self, X, y):
        super().set_params(objective=partial(XGBQuantile.quantile_loss,alpha = self.quant_alpha,delta = self.quant_delta,threshold = self.quant_thres,var = self.quant_var) )
        super().fit(X,y)
        return self
    
    def predict(self,X):
        return super().predict(X)
    
    def score(self, X, y):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1./score
        return score
        
    @staticmethod
    def quantile_loss(y_true,y_pred,alpha,delta,threshold,var):
        x = y_true - y_pred
        grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-  ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
        hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
    
        grad = (np.abs(x)<threshold )*grad - (np.abs(x)>=threshold )*(2*np.random.randint(2, size=len(y_true)) -1.0)*var
        hess = (np.abs(x)<threshold )*hess + (np.abs(x)>=threshold )
        return grad, hess
    
    @staticmethod
    def original_quantile_loss(y_true,y_pred,alpha,delta):
        x = y_true - y_pred
        grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
        hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
        return grad,hess

    
    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true-y_pred,alpha=alpha)
        score = np.sum(score)
        return score
    
    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha-1.0)*x*(x<0)+alpha*x*(x>=0)
    
    @staticmethod
    def get_split_gain(gradient,hessian,l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i])**2/(np.sum(hessian[:i])+l)+np.sum(gradient[i:])**2/(np.sum(hessian[i:])+l)-np.sum(gradient)**2/(np.sum(hessian)+l) )
        
        return np.array(split_gain)



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

    feature_names = ['Historical air quality (Bakke kirke-$PM_{2.5}$)',
                 'Historical air quality (Bakke kirke-$PM_{10}$)',
                 'Historical air quality (E6-Tiller-$PM_{2.5}$)',
                 'Historical air quality (E6-Tiller-$PM_{10}$)',
                 'Historical air quality (Elgeseter-$PM_{2.5}$)',
                 'Historical air quality (Elgeseter-$PM_{10}$)',
                 'Historical air quality (Torvet-$PM_{2.5}$)',
                 'Historical air quality (Torvet-$PM_{10}$)',
                 'Traffic (Elgeseter)',
                 'Traffic (Innherredsveien)',
                 'Traffic (Jernbanebrua)',
                 'Traffic (Strindheim)',
                 'Traffic (Bakkekirke)',
                 'Traffic (Moholtlia)',
                 'Traffic (Grillstadtunnelen)',
                 'Traffic (Oslovegen)',
                 'Air temperature (Voll)',
                 'Relative humidity (Voll)',
                 'Precipitation (Voll)',
                 'Air pressure (Voll)',
                 'Wind speed (Voll)',
                 'Wind direction (Voll)',
                 'Air Temperature (Sverreborg)',
                 'Relative humidity (Sverreborg)',
                 'Precipitation (Sverreborg)',
                 'Snow thickness (Sverreborg)',
                 'Duration of sunshine (Gloshaugen)',            
                 'Relative humidity (Lade)',
                 'Precipitation (Lade)',
                 'Air temperature (Lade)',   
                 'Street cleaning (Siemens to Sirkus)',
                 'Street cleaning (E6)',
                 'Street cleaning (706)',
                 'Street cleaning (Fv 6680)',
                 'Street cleaning (Sandgata)',
                 'Street cleaning (AvHaakon VII gate)']


    rows = len(stats['reg_cols'])
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 8))
    fig3, axs3 = plt.subplots(1, 1, figsize=(14, 7))
    fig, axs = plt.subplots(rows, 1, figsize=(20, 3.5*rows))
    axs2_counter=0
    test_index = pd.date_range(start=stats['start_test'], end=stats['end_test'], freq='H')[stats['sequence_length']:]
    xlim = [pd.to_datetime(stats['start_test']) - pd.to_timedelta(stats['sequence_length'], unit='H'), pd.to_datetime(stats['end_test'])]
    for i in range(0, rows):

        xgtrain = xgb.DMatrix(X_train, label=y_train[:, i])
        xgtest = xgb.DMatrix(X_test)
        param = {'max_depth':10, 'eta':0.15, 'gamma': 0.008, 'min_child_weight': 0.7, 'objective':'reg:squarederror' }
        num_round = 44
        bst = xgb.train(param, xgtrain, num_round)
        y_pred = bst.predict(xgtest)
        y_true = y_test[:, i]

        y_pred = rescale(y_pred, stats, i, task )
        y_true = rescale(y_true, stats, i, task)

        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        
        axs[i].plot(test_index, y_pred, color = 'r', label='Forecast')
        axs[i].plot(test_index, y_true, color = 'b', label='Observed value')
        axs[i].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}".format(stats['reg_cols'][i].split('_')[0], rmse))
        if '_pm25' in stats['reg_cols'][i]:
            axs[i].set_ylim(-1, 30)
            axs[i].set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
        else:
            axs[i].set_ylim(-1, 40)
            axs[i].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
        axs[i].set_xlim(xlim)
        axs[i].legend(loc="upper left")
        
        if stats['reg_cols'][i].split('_')[0] == 'Elgeseter':
            axs2[axs2_counter].plot(test_index, y_pred, color = 'r', label='XGBoost forecast')
            axs2[axs2_counter].plot(test_index, y_true, color = 'b', label='Observed value')
            axs2[axs2_counter].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}".format(stats['reg_cols'][i].split('_')[0], rmse))
            if '_pm25' in stats['reg_cols'][i]:
                axs2[axs2_counter].set_ylim(-1, 30)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{2.5}$(${\mu}g/m^3 $)")
            else:
                axs2[axs2_counter].set_ylim(-1, 40)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{10}$(${\mu}g/m^3 $)")
                
                bst.feature_names = feature_names
                feature_importance=bst.get_score(importance_type='weight')
                feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                bar_color = np.array(importance[::-1])
                bar_color = bar_color /bar_color.max()
                
                axs3.barh(y=features[::-1], width=importance[::-1], color=cmap(bar_color)[::-1])
                axs3.set_title('XGBoost Feature Importance')
                axs3.set_xlabel('Importance weight')
    #             axs3.set_ylabel('Importance weight')
                axs3.tick_params(axis='y', labelsize=10)
    #             axs3.tick_params(axis='x', labelrotation=75)
    #             axs3.set_xticklabels(axs3.get_xticklabels(), rotation=90, ha='right')
    #             axs3.xticks(rotation='vertical')
                
                
            axs2[axs2_counter].set_xlim(xlim)
            axs2[axs2_counter].legend(loc="upper left")
            axs2_counter += 1
        
    fig2.tight_layout()
    # fig2.savefig('../results/'+ save_name + '.pdf', bbox_inches='tight')

    fig3.tight_layout()
    # fig3.savefig('../results/feature_importance_'+ save_name + '.pdf', bbox_inches='tight')

    fig.tight_layout()
    # fig.savefig('../results/appendix/'+ save_name + '.pdf', bbox_inches='tight')


    save_name = 'xgboost_quantile'
    alpha = 0.95 
    rows = len(stats['reg_cols'])
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 7))
    fig, axs = plt.subplots(rows, 1, figsize=(20, 3.5*rows))
    axs2_counter=0
    test_index = pd.date_range(start=stats['start_test'], end=stats['end_test'], freq='H')[stats['sequence_length']:]
    xlim = [pd.to_datetime(stats['start_test']) - pd.to_timedelta(stats['sequence_length'], unit='H'), pd.to_datetime(stats['end_test'])]
    for i in range(0, rows):


        regressor = XGBRegressor(n_estimators=44,max_depth=5,reg_alpha=5, reg_lambda=1,gamma=0.08)
        y_pred = regressor.fit(X_train,y_train[:, i]).predict(X_test)

        regressor = XGBQuantile(n_estimators=44,max_depth = 15, reg_alpha =5.0,gamma = 0.08,reg_lambda =1.0 )     
        regressor.set_params(quant_alpha=1.-alpha,quant_delta=1.0,quant_thres=5.0,quant_var=3.2)
        pred_lower = regressor.fit(X_train,y_train[:, i]).predict(X_test)
        
        regressor.set_params(quant_alpha=alpha,quant_delta=1.0,quant_thres=6.0,quant_var = 4.2)
        pred_upper = regressor.fit(X_train,y_train[:, i]).predict(X_test)
        clear_output(wait=True)
        
        y_true = y_test[:, i]

        y_pred = rescale(y_pred, stats, i, task )
        y_true = rescale(y_true, stats, i, task)
        pred_upper = rescale(pred_upper , stats, i, task )
        pred_lower = rescale(pred_lower, stats, i, task)

        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        

        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        picp = prediction_interval_coverage_probability(y_true=y_true, y_lower=pred_lower, y_upper=pred_upper)
        mpiw = mean_prediction_interval_width(y_lower=pred_lower, y_upper=pred_upper)

        axs[i].fill_between(test_index, pred_lower, pred_upper,  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
        axs[i].plot(test_index, y_pred, color = 'r', label='Forecast')
        axs[i].plot(test_index, y_true, color = 'b', label='Observed value')
        axs[i].plot(test_index, pred_upper, color= 'k', linewidth=0.4)
        axs[i].plot(test_index, pred_lower, color= 'k', linewidth=0.4)
        axs[i].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw))
        if '_pm25' in stats['reg_cols'][i]:
            axs[i].set_ylim(-1, 30)
            axs[i].set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
        else:
            axs[i].set_ylim(-1, 40)
            axs[i].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
        axs[i].set_xlim(xlim)
        axs[i].legend(loc="upper left")

        #To use Elgeseter in the paper as representative examples
        if stats['reg_cols'][i].split('_')[0] == 'Elgeseter':
            axs2[axs2_counter].fill_between(test_index, pred_lower, pred_upper,  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
            axs2[axs2_counter].plot(test_index, y_pred, color = 'r', label='Forecast')
            axs2[axs2_counter].plot(test_index, y_true, color = 'b', label='Observed value')
            axs2[axs2_counter].plot(test_index, pred_upper, color= 'k', linewidth=0.4)
            axs2[axs2_counter].plot(test_index, pred_lower, color= 'k', linewidth=0.4)
            axs2[axs2_counter].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw))
            if '_pm25' in stats['reg_cols'][i]:
                axs2[axs2_counter].set_ylim(-1, 30)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{2.5}$(${\mu}g/m^3 $)")
            else:
                axs2[axs2_counter].set_ylim(-1, 40)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
            axs2[axs2_counter].set_xlim(xlim)
            axs2[axs2_counter].legend(loc="upper left")
            axs2_counter += 1


    fig2.tight_layout()
    # fig2.savefig('../results/'+ save_name + '.pdf', bbox_inches='tight')
    fig.tight_layout()
    # fig.savefig('../results/appendix/'+ save_name + '.pdf', bbox_inches='tight')


    save_name = 'gradient_boosting'
    alpha = 0.95 
    rows = len(stats['reg_cols'])
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 7))
    fig, axs = plt.subplots(rows, 1, figsize=(20, 3.5*rows))
    axs2_counter=0
    test_index = pd.date_range(start=stats['start_test'], end=stats['end_test'], freq='H')[stats['sequence_length']:]
    xlim = [pd.to_datetime(stats['start_test']) - pd.to_timedelta(stats['sequence_length'], unit='H'), pd.to_datetime(stats['end_test'])]
    for i in range(0, rows):

        regressor = regressor = GradientBoostingRegressor(n_estimators=44, max_depth=15,
                                    learning_rate=.1, min_samples_leaf=9,
                                    min_samples_split=9)
        y_pred = regressor.fit(X_train,y_train[:, i]).predict(X_test)

        regressor.set_params(loss='quantile', alpha=1.-alpha)    
        pred_lower = regressor.fit(X_train,y_train[:, i]).predict(X_test)
        
        regressor.set_params(loss='quantile', alpha=alpha)
        pred_upper = regressor.fit(X_train,y_train[:, i]).predict(X_test)
        
        clear_output(wait=True)
        
        y_true = y_test[:, i]

        y_pred = rescale(y_pred, stats, i, task )
        y_true = rescale(y_true, stats, i, task)
        pred_upper = rescale(pred_upper , stats, i, task )
        pred_lower = rescale(pred_lower, stats, i, task)

        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        

        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        picp = prediction_interval_coverage_probability(y_true=y_true, y_lower=pred_lower, y_upper=pred_upper)
        mpiw = mean_prediction_interval_width(y_lower=pred_lower, y_upper=pred_upper)

        axs[i].fill_between(test_index, pred_lower, pred_upper,  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
        axs[i].plot(test_index, y_pred, color = 'r', label='Forecast')
        axs[i].plot(test_index, y_true, color = 'b', label='Observed value')
        axs[i].plot(test_index, pred_upper, color= 'k', linewidth=0.4)
        axs[i].plot(test_index, pred_lower, color= 'k', linewidth=0.4)
        axs[i].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw))
        if '_pm25' in stats['reg_cols'][i]:
            axs[i].set_ylim(-1, 30)
            axs[i].set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
        else:
            axs[i].set_ylim(-1, 40)
            axs[i].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
        axs[i].set_xlim(xlim)
        axs[i].legend(loc="upper left")

        #To use Elgeseter in the paper as representative examples
        if stats['reg_cols'][i].split('_')[0] == 'Elgeseter':
            axs2[axs2_counter].fill_between(test_index, pred_lower, pred_upper,  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
            axs2[axs2_counter].plot(test_index, y_pred, color = 'r', label='Forecast')
            axs2[axs2_counter].plot(test_index, y_true, color = 'b', label='Observed value')
            axs2[axs2_counter].plot(test_index, pred_upper, color= 'k', linewidth=0.4)
            axs2[axs2_counter].plot(test_index, pred_lower, color= 'k', linewidth=0.4)
            axs2[axs2_counter].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw))
            if '_pm25' in stats['reg_cols'][i]:
                axs2[axs2_counter].set_ylim(-1, 30)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{2.5}$(${\mu}g/m^3 $)")
            else:
                axs2[axs2_counter].set_ylim(-1, 40)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
            axs2[axs2_counter].set_xlim(xlim)
            axs2[axs2_counter].legend(loc="upper left")
            axs2_counter += 1


    fig2.tight_layout()
    # fig2.savefig('../results/'+ save_name + '.pdf', bbox_inches='tight')
    fig.tight_layout()
    # fig.savefig('../results/appendix/'+ save_name + '.pdf', bbox_inches='tight')