import os
import numpy as np
import pandas as pd
import torch
from scipy import special
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)

from argparse import Namespace
from probabilistic_forecast.utils.data_utils import data_loader
from probabilistic_forecast.utils.torch_utils import torch_loader
from probabilistic_forecast.utils.plot_utils import plot_results

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'grid.linestyle': '--'})


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

def rescale(y, stats):
    y = y * stats['y_reg_std'] + stats['y_reg_mean']
    y = np.exp(y)-1
    return y


def plot_regression(target_test, mixture_mean, mixture_var, stats):

    NLL_criterion = torch.nn.GaussianNLLLoss(full=True, reduction='mean')
    Standard_scores = {'z_90': 1.64, 'z_95':1.96, 'z_99': 2.58}
    pred_upper = mixture_mean + Standard_scores['z_95']*np.sqrt(mixture_var)
    pred_lower = mixture_mean - Standard_scores['z_95']*np.sqrt(mixture_var)

    pred_mean = rescale(mixture_mean, stats)
    pred_upper = rescale(pred_upper, stats)
    pred_lower = rescale(pred_lower, stats)
    y_true = rescale(target_test, stats)        

    rows = len(stats['reg_cols'])
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 7))
    fig, axs = plt.subplots(rows, 1, figsize=(20, 3.5*rows))
    axs2_counter=0
    test_index = pd.date_range(start=stats['start_test'], end=stats['end_test'], freq='H')[stats['historical_sequence_length']:]
    xlim = [pd.to_datetime(stats['start_test']) - pd.to_timedelta(stats['historical_sequence_length'], unit='H'), pd.to_datetime(stats['end_test'])]
    for i in range(0, rows):

        crps = crps_gaussian(target_test[:, i], mixture_mean[:, i], np.sqrt(mixture_var[:, i]))
        nll = NLL_criterion(torch.tensor(target_test[:, i]), torch.tensor(mixture_mean[:, i]), torch.tensor(mixture_var[:, i])).item()

        rmse = root_mean_squared_error(y_true=y_true[:, i], y_pred=pred_mean[:, i])
        picp = prediction_interval_coverage_probability(y_true=y_true[:, i], y_lower=pred_lower[:, i], y_upper=pred_upper[:, i])
        mpiw = mean_prediction_interval_width(y_lower=pred_lower[:, i], y_upper=pred_upper[:, i])

        axs[i].fill_between(test_index, pred_lower[:, i], pred_upper[:, i],  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
        axs[i].plot(test_index, pred_mean[:, i], color = 'r', label='Forecast')
        axs[i].plot(test_index, y_true[:, i], color = 'b', label='Observed value')
        axs[i].plot(test_index, pred_upper[:, i], color= 'k', linewidth=0.4)
        axs[i].plot(test_index, pred_lower[:, i], color= 'k', linewidth=0.4)
        axs[i].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f}, CRPS ={4:0.2f}, NLL = {5:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw, crps, nll))
        print("Monitoring Station ({0}): &{1:0.2f}&{2:0.2f}&{3:0.2f}&{4:0.2f}&{5:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw, crps, nll))
        if '_pm25' in stats['reg_cols'][i]:
            axs[i].set_ylim(-1, 30)
            axs[i].set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
        else:
            axs[i].set_ylim(-1, 40)
            axs[i].set_ylabel("Air pollutant $PM_{10}$(${\mu}g/m^3 $)")
        axs[i].set_xlim(xlim)
        axs[i].legend(loc="upper left")

        #To use Elgeseter in the paper as representative examples
        if stats['reg_cols'][i].split('_')[0] == 'Elgeseter':
            axs2[axs2_counter].fill_between(test_index, pred_lower[:, i], pred_upper[:, i],  alpha=.3, fc='red', ec='None', label='95% Prediction interval')
            axs2[axs2_counter].plot(test_index, pred_mean[:, i], color = 'r', label='Forecast')
            axs2[axs2_counter].plot(test_index, y_true[:, i], color = 'b', label='Observed value')
            axs2[axs2_counter].plot(test_index, pred_upper[:, i], color= 'k', linewidth=0.4)
            axs2[axs2_counter].plot(test_index, pred_lower[:, i], color= 'k', linewidth=0.4)
            axs2[axs2_counter].set_title("Monitoring Station ({0}): RMSE = {1:0.2f}, PICP = {2:0.2f}, MPIW = {3:0.2f}, CRPS ={4:0.2f}, NLL = {5:0.2f} ".format(stats['reg_cols'][i].split('_')[0], rmse, picp, mpiw, crps, nll))
            if '_pm25' in stats['reg_cols'][i]:
                axs2[axs2_counter].set_ylim(-1, 30)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
            else:
                axs2[axs2_counter].set_ylim(-1, 40)
                axs2[axs2_counter].set_ylabel("Air pollutant $PM_{10}$ (${\mu}g/m^3 $)")
            axs2[axs2_counter].set_xlim(xlim)
            axs2[axs2_counter].legend(loc="upper left")
            axs2_counter += 1

    fig2.tight_layout()
#     fig2.savefig(fig_save_name + '.jpg', bbox_inches='tight')
    fig.tight_layout()
#     fig.savefig(fig_save_name + '_all_stations.jpg', bbox_inches='tight')


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from probabilistic_forecast.utils.torch_utils import get_device



class ConvLSTM():

    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net_arch = [128, 32]
        self.task = args.task
        self.device = get_device()

        self.network = Network(self.input_dim, self.output_dim, self.net_arch, self.task, self.device)
        self.network.to(self.device)
        if self.task == "regression":
            self.criterion = torch.nn.GaussianNLLLoss(full=True, reduction='sum')
        elif self.task == "classification":
            self.criterion = nn.BCELoss(reduction='sum')


    def get_loss(self, output, target):
        if self.task == "regression":
            return self.criterion(output[0], target, output[1]) 
        elif self.task == "classification":
            return self.criterion(output, target)


    def train(self, train_loader, n_epochs, batch_size, stats, pre_trained_dir, Nbatches, adversarial_training=True):
        print('Training {} model {} adversarial training. Task: {}'.format(type(self).__name__, 
            'with' if adversarial_training else 'without', self.task))

        learning_rate=1e-2
        weight_decay = 1e-3
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.999)

        if adversarial_training:

            delta = torch.zeros([batch_size, stats['historical_sequence_length'], self.input_dim]).to(self.device)
            X_train_max= torch.tensor(stats['X_train_max'])
            X_train_max= X_train_max.to(self.device)

            X_train_min= torch.tensor(stats['X_train_min'])
            X_train_min= X_train_min.to(self.device)

            if self.task == 'regression':
                clip_eps = 1.0 / stats['X_train_max'].max()
                fgsm_step = 1.0 / stats['X_train_max'].max()
            elif self.task == 'classification':
                clip_eps = 0.5 / stats['X_train_max'].max()
                fgsm_step = 0.5 / stats['X_train_max'].max()

            n_repeats = 4
            n_epochs = int(n_epochs / n_repeats)
        
        self.network.train()
        loss_history, lr_history = [], []
        for epoch in range(1, n_epochs+1 ):
            epoch_loss =[]
            for _ , (features , target) in enumerate(train_loader):
                features  = features.to(self.device)
                target = target.to(self.device)
                
                if adversarial_training:
                    for _ in range(n_repeats):
                        delta_batch = delta[0:features.size(0)]
                        delta_batch.requires_grad = True
                        adv_features = features + delta_batch
                        adv_features.clamp_(X_train_min, X_train_max)
                        output = self.network(adv_features)
                        loss = self.get_loss(output, target)
                        loss.backward()
                        optimizer.step()        
                        optimizer.zero_grad()
                        pert = fgsm_step * torch.sign(delta_batch.grad)
                        delta[0:features.size(0)] += pert.data
                        delta.clamp_(-clip_eps, clip_eps)
                        epoch_loss.append(loss.item())
                else:
                    output = self.network(features)
                    loss = self.get_loss(output, target)
                    loss.backward()
                    optimizer.step()        
                    optimizer.zero_grad()
                    epoch_loss.append(loss.item())

            lr_history.append(optimizer.param_groups[0]['lr'])
            lr_scheduler.step()
            loss_history.append(np.mean(epoch_loss))

            if epoch % 10 == 0:
                print("Epoch: {0:0.3g}, NLL: {1:0.3g},  lr: {2:0.3g}".format(epoch,loss_history[-1], lr_history[-1]), end='\r')


        pre_trained_dir = os.path.join(pre_trained_dir, type(self).__name__)
        os.makedirs(pre_trained_dir , exist_ok=True)
        model_save_name = pre_trained_dir + '/trained_network_' + self.task + ('_adv.pt' if adversarial_training else '.pt')
        fig_save_name = pre_trained_dir + '/training_curve_' +self.task + ('_adv.pdf' if adversarial_training else '.pdf')
        torch.save(self.network.state_dict(), model_save_name)
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        axs[0].plot(loss_history)
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel("NLL Loss")
        axs[1].plot(lr_history)
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel("Learning rate")
        axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        fig.tight_layout()
        
#         plot_training_curve(loss_history, lr_history, fig_save_name)
        

    def evaluate(self, test_loader, n_samples, pre_trained_dir, adversarial_training=True):
        print('Evaluating a pretrained {} model {} adversarial training. Task: {}'.format(type(self).__name__, 
            'with' if adversarial_training else 'without', self.task))
        pre_trained_dir = os.path.join(pre_trained_dir, type(self).__name__)
        model_save_name = pre_trained_dir + '/trained_network_'+ self.task + ('_adv.pt' if adversarial_training else '.pt')
        self.network.load_state_dict(torch.load(model_save_name))
        self.network.eval()
        if self.task =='regression':
            samples_mean, samples_var = [], []
            for _ in range(n_samples):
                pred_mean_set, pred_var_set = [], []
                for _, (features , _ ) in enumerate(test_loader):
                    features = features.to(self.device)
                    pred_mean, pred_var= self.network(features)
                    pred_mean_set.append(pred_mean.detach().cpu().numpy())
                    pred_var_set.append(pred_var.detach().cpu().numpy())
                pred_mean_i, pred_var_i =  np.concatenate(pred_mean_set, axis=0), np.concatenate(pred_var_set, axis=0)
                samples_mean.append(pred_mean_i)
                samples_var.append(pred_var_i)
            samples_mean = np.array(samples_mean)
            samples_var = np.array(samples_var)
            mixture_mean = np.mean(samples_mean, axis=0)
            mixture_var = np.mean(samples_var + np.square(samples_mean), axis=0) - np.square(mixture_mean)
            target_test  = self.get_target_test(test_loader)
            return target_test, mixture_mean, mixture_var

        elif self.task =='classification':
            samples = []
            for _ in range(n_samples):
                pred = []
                for _, (features , _ ) in enumerate(test_loader):
                    features  = features.to(self.device)
                    output= self.network(features)
                    pred.append(output.detach().cpu().numpy())
                pred_i = np.concatenate(pred, axis=0)
                samples.append(pred_i)
            samples = np.array(samples)  
            target_test  = self.get_target_test(test_loader)
            return target_test, samples

    def get_target_test(self, test_loader):
        target_set = []
        for _ , ( _ , target) in enumerate(test_loader):
            target_set.append(target.numpy())
        return np.concatenate(target_set, axis=0)





class Network(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, task, device):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_probability = 0.5
        
        self.task=task
        self.device = device

        self.hidden_size_1 = net_arch[0] 
        self.hidden_size_2 = net_arch[1]
        self.stacked_layers = 2

        self.lstm1 = nn.LSTM(self.input_dim, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)

        
        if self.task == 'regression':
            self.fc = nn.Linear(5280, 2*self.output_dim)
            self.Softplus= nn.Softplus()
        elif self.task == 'classification':
            self.fc = nn.Linear(5280, self.output_dim)
            self.Sigmoid= nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        

                               
    def forward(self, x):
        
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, None, :, :]
        output = self.conv(output)
        output = F.relu(output)
        output = F.max_pool2d(output, 2)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = torch.flatten(output, 1)
        out = self.fc(output)
 
        if self.task == 'regression':
            mean = out[:, :self.output_dim]
            # The variance should always be positive (softplus) and la
            variance = self.Softplus(out[:, self.output_dim:])+ 1e-06 
            return mean, variance

        elif self.task == 'classification':
            prob = self.Sigmoid(out)
            return prob

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)).to(self.device)
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)).to(self.device)
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)).to(self.device)
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)).to(self.device)
        return hidden_state, cell_state 


if __name__ == '__main__':


    task='regression'
    sequential= True
    args_dict = {'adversarial_training': False, 'batch_size':128, 'data_dir':'../dataset', 'end_test':'2020-02-01',
                'end_train':'2019-12-31', 'forecast_horizon':24, 'historical_sequence_length':24, 'mode':'evaluate',
                'n_epochs':500, 'n_samples':1000, 'plots_dir':'../plots', 'pretrained_dir':'../pretrained', 
                'start_test':'2020-01-01', 'start_train':'2019-01-01', 'task':task}
    args = Namespace(**args_dict)
    X_train, y_train, X_test, y_test, stats =data_loader(args.data_dir,
                                                    task=args.task, 
                                                    historical_sequence_length=args.historical_sequence_length, 
                                                    forecast_horizon=args.forecast_horizon, 
                                                    start_train=args.start_train,
                                                    end_train=args.end_train,
                                                    start_test=args.start_test,
                                                    end_test=args.end_test)
    train_loader, test_loader = torch_loader(X_train, y_train, X_test, y_test, args.historical_sequence_length, args.batch_size, sequential)
    input_dim = X_train.shape[-1] if sequential else X_train.shape[-1] * args.historical_sequence_length
    output_dim = y_train.shape[-1]

    model = ConvLSTM(input_dim, output_dim, args)
    Nbatches = X_train.shape[0]/args.batch_size
    model.train(train_loader, args.n_epochs, args.batch_size, stats, args.pretrained_dir, Nbatches, args.adversarial_training)
    results = model.evaluate(test_loader, args.n_samples, args.pretrained_dir, args.adversarial_training)
    plot_regression(results[0], results[1], results[2], stats)