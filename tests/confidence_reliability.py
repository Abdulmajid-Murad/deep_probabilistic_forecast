import os
from argparse import Namespace

from probabilistic_forecast.nn_mc import NN_MC
from probabilistic_forecast.ensemble import Deep_Ensemble
from probabilistic_forecast.swag import SWAG
from probabilistic_forecast.lstm_mc import LSTM_MC
from probabilistic_forecast.bnn import BNN
from probabilistic_forecast.gnn_mc import GNN_MC

from probabilistic_forecast.utils.data_utils import data_loader
from probabilistic_forecast.utils.torch_utils import torch_loader
from probabilistic_forecast.utils.plot_utils import plot_confidence_reliability

def evaluate_single_model(args):


    models_types= {'NN_MC': {'model_class':NN_MC, 'sequential': False }, 
            'Deep_Ensemble':{'model_class':Deep_Ensemble, 'sequential': False }, 
            'SWAG':{'model_class':SWAG, 'sequential': False }, 
            'LSTM_MC': {'model_class':LSTM_MC, 'sequential': True }, 
            'BNN': {'model_class':BNN, 'sequential': False}, 
            'GNN_MC':{'model_class':GNN_MC, 'sequential': True }
        }
    model_class = models_types[args.model]['model_class']
    sequential = models_types[args.model]['sequential']
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

    model = model_class(input_dim, output_dim, args)

    if args.model == 'SWAG':
        # BatchNorm buffers update using train dataset.
        results = model.evaluate(test_loader, args.n_samples,  args.pretrained_dir,  train_loader,  args.adversarial_training)
    else:
        results = model.evaluate(test_loader, args.n_samples, args.pretrained_dir, args.adversarial_training)

    return results, stats


def evaluate_all_models(task):
    args_dict = {'adversarial_training': False, 'batch_size':128, 'data_dir':'../dataset', 'end_test':'2020-02-01',
            'end_train':'2019-12-31', 'forecast_horizon':24, 'historical_sequence_length':24, 'mode':'evaluate',
            'n_epochs':2000, 'n_samples':1000, 'plots_dir':'../plots', 'pretrained_dir':'../pretrained', 
            'start_test':'2020-01-01', 'start_train':'2019-01-01', 'task':task}

    resutls_all_models = {}
    models = ['BNN', 'NN_MC', 'Deep_Ensemble', 'LSTM_MC', 'GNN_MC', 'SWAG' ]
    for model in models:
        args_dict['model'] = model
        args = Namespace(**args_dict)
        results, stats = evaluate_single_model(args)
        results_model={}
        if task == 'regression':
            results_model['target_test'], results_model['mixture_mean'], results_model['mixture_var'], results_model['stats'] = results[0], results[1], results[2], stats
        elif task == 'classification':
            results_model['target_test'], results_model['samples'], results_model['stats'] = results[0], results[1], stats
        resutls_all_models[model] = results_model
    plot_confidence_reliability(resutls_all_models, args)


if __name__ == "__main__":
    evaluate_all_models(task='regression')
    evaluate_all_models(task='classification')