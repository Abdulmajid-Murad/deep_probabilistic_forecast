import os
import numpy as np
import pandas as pd
import torch
from scipy import special
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)

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

def nll_gaussian(x, mu, sig):
    exponent = -0.5*(x - mu)**2/sig**2
    log_coeff = np.log(sig) - 0.5*np.log(2*np.pi)
    return - (log_coeff + exponent).mean()

def plot_confidence_reliability(resutls_all_models, args):
    plots_dir = os.path.join(args.plots_dir)
    os.makedirs(plots_dir , exist_ok=True)
    fig_save_name = plots_dir +'/'+args.task + ('_confidence_reliability_adv' if args.adversarial_training else '_confidence_reliability')
    if args.task == "regression":
        plot_confidence_reliability_regression(resutls_all_models, fig_save_name)
    elif args.task == "classification":
        plot_confidence_reliability_classification(resutls_all_models, fig_save_name)

def plot_results(results, args, stats):

    plots_dir = os.path.join(args.plots_dir, args.model)
    os.makedirs(plots_dir , exist_ok=True)
    fig_save_name = plots_dir +'/'+args.task + ('_adv' if args.adversarial_training else '')
    if args.task == "regression":
        plot_regression(results[0], results[1], results[2], stats, fig_save_name)
    elif args.task == "classification":
        plot_classification(results[0], results[1], stats, fig_save_name)

def plot_confidence_reliability_regression(resutls_all_models, fig_save_name, extension='.jpg'):
    
    Standard_scores = {'z_90': 1.64, 'z_95':1.96, 'z_99': 2.58}
    all_max_ciw  = []
    for key, value in resutls_all_models.items():
        target_test,  mixture_mean, mixture_var, stats = value['target_test'], value['mixture_mean'],value['mixture_var'], value['stats']
        upper = rescale(mixture_mean + Standard_scores['z_95']*np.sqrt(mixture_var), stats)
        lower = rescale(mixture_mean - Standard_scores['z_95']*np.sqrt(mixture_var), stats)
        ciw = upper - lower
        max_ciw = ciw.max(axis=0)
        all_max_ciw.append(max_ciw)      
    all_max_ciw = np.array(all_max_ciw)
    max_ciw = all_max_ciw.max(axis=0)

    confidence_thresholds = np.arange(0, 1.0, 0.05)

    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
    fig, axs = plt.subplots(4, 4, figsize=(20, 3.5*4))

    for key, value in resutls_all_models.items():
        target_test, mixture_mean, mixture_var, stats = value['target_test'], value['mixture_mean'],value['mixture_var'], value['stats']
        upper = rescale(mixture_mean + Standard_scores['z_95']*np.sqrt(mixture_var), stats)
        lower = rescale(mixture_mean - Standard_scores['z_95']*np.sqrt(mixture_var), stats)
        ciw = upper - lower
        confidence = 1 - (ciw / max_ciw)

        y_pred = rescale(mixture_mean, stats)
        y_true = rescale(target_test, stats)  
        absolute_error =  np.absolute((y_true - y_pred))
        rows = len(stats['reg_cols'])
    #     axs2_counter=0
        for i in range(0, rows):
            predict_confidence = confidence[:, i]
            error = absolute_error[:, i]

            shape = (len(confidence_thresholds),)
            losses = np.zeros(shape)
            counts = np.zeros(shape)
            for idx, thresh in enumerate(confidence_thresholds):
                mask = predict_confidence >= thresh
                counts[idx] = mask.sum(-1)
                losses[idx] = np.ma.masked_array(error, mask=~mask).sum()

            if '_pm25' in stats['reg_cols'][i]:
                particulate = "PM2.5"
                idx = 0
            else:
                particulate = "PM10"
                idx = 1

            axs[int(i/2)][2*(i%2)].plot(confidence_thresholds, losses, label=key)
            axs[int(i/2)][2*(i%2)].set_xlabel(r"Confidence Threshold $\tau$")
            axs[int(i/2)][2*(i%2)].set_ylabel(r"Loss for confidence $ \geq \tau$")
            axs[int(i/2)][2*(i%2)].set_title("Monitoring Station({0} - {1})".format(stats['reg_cols'][i].split('_')[0],particulate))
            axs[int(i/2)][2*(i%2)].legend(loc='lower left')
            axs[int(i/2)][2*(i%2)+1].plot(confidence_thresholds, counts, label=key)
            axs[int(i/2)][2*(i%2)+1].set_xlabel(r"Confidence Threshold $\tau$")
            axs[int(i/2)][2*(i%2)+1].set_ylabel("Count for " + r"confidence $\geq \tau$")
            axs[int(i/2)][2*(i%2)+1].set_title("Monitoring Station({0} - {1})".format(stats['reg_cols'][i].split('_')[0],particulate))
            axs[int(i/2)][2*(i%2)+1].legend(loc='lower left')

            if stats['reg_cols'][i].split('_')[0] == 'Elgeseter':
                axs2[idx][0].plot(confidence_thresholds, losses, label=key)
                axs2[idx][0].set_xlabel(r"Confidence Threshold $\tau$")
                axs2[idx][0].set_ylabel(r"Loss for confidence $ \geq \tau$")
                axs2[idx][0].set_title("Monitoring Station({0} - {1})".format(stats['reg_cols'][i].split('_')[0],particulate))
                axs2[idx][0].legend(loc='lower left')
                axs2[idx][1].plot(confidence_thresholds, counts, label=key)
                axs2[idx][1].set_xlabel(r"Confidence Threshold $\tau$")
                axs2[idx][1].set_ylabel("Count for " + r"confidence $\geq \tau$")
                axs2[idx][1].set_title("Monitoring Station({0} - {1})".format(stats['reg_cols'][i].split('_')[0],particulate))
                axs2[idx][1].legend(loc='lower left')
            
    fig2.tight_layout()
    fig2.savefig(fig_save_name + extension, bbox_inches='tight')
    fig.tight_layout()
    fig.savefig(fig_save_name + '_all_stations' + extension, bbox_inches='tight')

def plot_confidence_reliability_classification(resutls_all_models, fig_save_name, extension='.jpg'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from IPython.display import clear_output
    import seaborn as sns
    sns.set_theme()
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axs = plt.subplots(4, 4, figsize=(20, 3.5*4))

    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
    confidence_thresholds = np.arange(0, 1.01, 0.05)

    for key, value in resutls_all_models.items():
        
        target_test, samples, stats = value['target_test'], value['samples'], value['stats']
        
        y_prob = np.mean(samples, axis=0)
        y_pred = np.rint(y_prob).astype(int) 
        y_true = target_test.astype(int)   
        p5, p95 = np.quantile(samples, [0.05, 0.95], axis=0)
        confidence = 1 - (p95 - p5)
        rows = len(stats['thre_cols'])

        for i in range(0, rows):
            labels = y_true[:, i]
            predict_class = y_pred[:, i]
            predict_confidence = confidence[:, i]
            
            shape = (len(confidence_thresholds),)
            incorrect_predictions = np.zeros(shape)
            counts = np.zeros(shape)
            error = np.not_equal(predict_class, labels)
            for idx, thresh in enumerate(confidence_thresholds):
                mask = predict_confidence >= thresh
                counts[idx] = mask.sum(-1)
                incorrect_predictions[idx] = np.ma.masked_array(error, mask=~mask).sum()
                               
            if '_pm25' in stats['thre_cols'][i]:
                particulate = "PM2.5"
                idx = 0
            else:
                particulate = "PM10"
                idx = 1

            axs[int(i/2)][2*(i%2)].plot(confidence_thresholds, incorrect_predictions, label=key)
            axs[int(i/2)][2*(i%2)].set_xlabel(r"Confidence Threshold $\tau$")
            axs[int(i/2)][2*(i%2)].set_ylabel(r"Loss for confidence $ \geq \tau$")
            axs[int(i/2)][2*(i%2)].set_title("Monitoring Station({0} - {1})".format(stats['thre_cols'][i].split('_')[0],particulate))
            axs[int(i/2)][2*(i%2)].legend(loc='lower left')
            axs[int(i/2)][2*(i%2)+1].plot(confidence_thresholds, counts, label=key)
            axs[int(i/2)][2*(i%2)+1].set_xlabel(r"Confidence Threshold $\tau$")
            axs[int(i/2)][2*(i%2)+1].set_ylabel("Count for " + r"confidence $\geq \tau$")
            axs[int(i/2)][2*(i%2)+1].set_title("Monitoring Station({0} - {1})".format(stats['thre_cols'][i].split('_')[0],particulate))
            axs[int(i/2)][2*(i%2)+1].legend(loc='lower left')
        
            if stats['thre_cols'][i].split('_')[0] == 'Elgeseter':
                axs2[idx][0].plot(confidence_thresholds, incorrect_predictions, label=key)
                axs2[idx][0].set_xlabel(r"Confidence Threshold $\tau$")
                axs2[idx][0].set_ylabel(r"Loss for confidence $ \geq \tau$")
                axs2[idx][0].set_title("Monitoring Station({0} - {1})".format(stats['thre_cols'][i].split('_')[0],particulate))
                axs2[idx][0].legend(loc='lower left')
                axs2[idx][1].plot(confidence_thresholds, counts, label=key)
                axs2[idx][1].set_xlabel(r"Confidence Threshold $\tau$")
                axs2[idx][1].set_ylabel("Count for " + r"confidence $\geq \tau$")
                axs2[idx][1].set_title("Monitoring Station({0} - {1})".format(stats['thre_cols'][i].split('_')[0],particulate))
                axs2[idx][1].legend(loc='lower left')
                
    fig2.tight_layout()
    fig2.savefig(fig_save_name + extension, bbox_inches='tight')
    fig.tight_layout()
    fig.savefig(fig_save_name + '_all_stations' + extension, bbox_inches='tight')


def plot_regression(target_test, mixture_mean, mixture_var, stats, fig_save_name):

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
    fig2.savefig(fig_save_name + '.jpg', bbox_inches='tight')
    fig.tight_layout()
    fig.savefig(fig_save_name + '_all_stations.jpg', bbox_inches='tight')


def plot_classification(target_test, samples, stats, fig_save_name):

    bce_criterion = torch.nn.BCELoss(reduction='mean')
    p5, p95 = np.quantile(samples, [0.05, 0.95], axis=0)
    y_prob = np.mean(samples, axis=0)
    y_pred = np.rint(y_prob)
    y_true = target_test.astype(int)        


    rows = len(stats['thre_cols'])
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 7))
    fig, axs = plt.subplots(rows, 1, figsize=(20, 3.5*rows))
    axs2_counter=0
    test_index = pd.date_range(start=stats['start_test'], end=stats['end_test'], freq='H')[stats['historical_sequence_length']:]
    ylabel_format = '{:,.0f}%'
    xlim = [pd.to_datetime(stats['start_test']) - pd.to_timedelta(stats['historical_sequence_length'], unit='H'), pd.to_datetime(stats['end_test'])]
    
    for i in range(0, rows):

        brier_score = brier_score_loss(y_true=y_true[:, i], y_prob=y_prob[:, i] , pos_label=1)
        Precision=precision_score(y_true=y_true[:, i], y_pred=y_pred[:, i])
        Recall=recall_score(y_true=y_true[:, i], y_pred=y_pred[:, i])
        F1=f1_score(y_true=y_true[:, i], y_pred=y_pred[:, i])
        bce = bce_criterion(torch.tensor(y_prob[:, i]),torch.tensor(target_test[:, i])).item()
    
        y = y_true[:, i]*100
        x= test_index
        x_filt = x[y>0]
        y_filt= y[y>0]

        axs[i].scatter(x_filt, y_filt, color = 'b', label='Observed threshold \n exceedance event')
        axs[i].fill_between(test_index, p5[:, i]*100, p95[:, i]*100,  alpha=.3, fc='r', ec='None', label='95% Prediction interval')
        axs[i].plot(test_index, y_prob[:, i]*100, color = 'r',linewidth=0.8,  label='Forecast threshold\n exceedance probability')
        axs[i].plot(test_index, p5[:, i]*100, color= 'k', linewidth=0.4)
        axs[i].plot(test_index, p95[:, i]*100, color= 'k', linewidth=0.4)
        if '_pm25' in stats['thre_cols'][i]:
            particulate = "PM2.5"
        else:
            particulate = "PM10"
        axs[i].set_title("Monitoring Station({0} - {1}): Brier = {2:0.2f}, Precision = {3:0.2f}, Recall= {4:0.2f}, F1 = {5:0.2f}, CE = {6:0.2f}".format(stats['thre_cols'][i].split('_')[0],particulate, brier_score, Precision, Recall, F1, bce))
        print("Monitoring Station({0} - {1}): &{2:0.2f}&{3:0.2f}&{4:0.2f}&{5:0.2f}&{6:0.2f}".format(stats['thre_cols'][i].split('_')[0],particulate, brier_score, Precision, Recall, F1, bce))
        # axs[i].set_ylabel('Threshold Exceedance Prob.')
        axs[i].set_xlim(xlim)
        axs[i].legend(loc="upper left")

        ticks_loc = axs[i].get_yticks().tolist()
        axs[i].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axs[i].set_yticklabels([ylabel_format.format(x) for x in ticks_loc])
        #To use Elgeseter in the paper as representative examples
        if stats['thre_cols'][i].split('_')[0] == 'Elgeseter':
            axs2[axs2_counter].scatter(x_filt, y_filt, color = 'b', label='Observed threshold \n exceedance event')
            axs2[axs2_counter].fill_between(test_index, p5[:, i]*100, p95[:, i]*100,  alpha=.3, fc='r', ec='None', label='95% Prediction interval')
            axs2[axs2_counter].plot(test_index, y_prob[:, i]*100, color = 'r',linewidth=0.8,  label='Forecast threshold\n exceedance probability')
            axs2[axs2_counter].plot(test_index, p5[:, i]*100, color= 'k', linewidth=0.4)
            axs2[axs2_counter].plot(test_index, p95[:, i]*100, color= 'k', linewidth=0.4)
            axs2[axs2_counter].set_title("Monitoring Station({0} - {1}): Brier = {2:0.2f}, Precision = {3:0.2f}, Recall= {4:0.2f}, F1 = {5:0.2f}, CE = {6:0.2f}".format(stats['thre_cols'][i].split('_')[0],particulate, brier_score, Precision, Recall, F1, bce))
            # axs2[axs2_counter].set_ylabel('Threshold Exceedance Prob.')
            axs2[axs2_counter].set_xlim(xlim)
            axs2[axs2_counter].legend(loc="upper left")

            ticks_loc = axs2[axs2_counter].get_yticks().tolist()
            axs2[axs2_counter].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            axs2[axs2_counter].set_yticklabels([ylabel_format.format(x) for x in ticks_loc])
            axs2_counter += 1

    fig2.tight_layout()
    fig2.savefig(fig_save_name + '.jpg', bbox_inches='tight')
    fig.tight_layout()
    fig.savefig(fig_save_name + '_all_stations.jpg', bbox_inches='tight')

def plot_training_curve(loss_history, lr_history, fig_save_name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(loss_history)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel("NLL Loss")
    axs[1].plot(lr_history)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel("Learning rate")
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(fig_save_name, bbox_inches='tight')

def plot_training_curve_ensemble(ensemble_loss_history, ensemble_lr_history, fig_save_name):
    ensemble_size = len(ensemble_loss_history)
    fig, axs = plt.subplots(ensemble_size, 2, figsize=(12, 3*ensemble_size))
    model_idx=0
    for model_idx, (loss_history, lr_history) in enumerate(zip(ensemble_loss_history, ensemble_lr_history)):
        axs[model_idx][0].plot(loss_history)
        axs[model_idx][0].set_xlabel('Epochs')
        axs[model_idx][0].set_ylabel("NLL Loss")
        axs[model_idx][1].plot(lr_history)
        axs[model_idx][1].set_xlabel('Epochs')
        axs[model_idx][1].set_ylabel("Learning rate")
        axs[model_idx][1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        model_idx += 1
    fig.tight_layout()
    fig.savefig(fig_save_name, bbox_inches='tight')


def plot_training_curve_bnn(nll_history, kl_history, lr_history, fig_save_name):
    fig, axs = plt.subplots(1, 3, figsize=(21, 4))
    axs[0].plot(nll_history)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel("NLL Loss")
    axs[1].plot(kl_history)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel("KLL Loss")

    axs[2].plot(lr_history)
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel("Learning rate")
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(fig_save_name, bbox_inches='tight')

def rescale(y, stats):
    y = y * stats['y_reg_std'] + stats['y_reg_mean']
    y = np.exp(y)-1
    return y