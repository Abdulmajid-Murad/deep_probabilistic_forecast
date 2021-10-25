import os
import argparse

from probabilistic_forecast.nn_mc import NN_MC
from probabilistic_forecast.ensemble import Deep_Ensemble
from probabilistic_forecast.swag import SWAG
from probabilistic_forecast.lstm_mc import LSTM_MC
from probabilistic_forecast.bnn import BNN
from probabilistic_forecast.gnn_mc import GNN_MC
from probabilistic_forecast.nn_standard import NN_Standard

from probabilistic_forecast.utils.data_utils import data_loader
from probabilistic_forecast.utils.torch_utils import torch_loader
from probabilistic_forecast.utils.plot_utils import plot_results


def run(args):


    models_types= {'NN_MC': {'model_class':NN_MC, 'sequential': False }, 
            'Deep_Ensemble':{'model_class':Deep_Ensemble, 'sequential': False }, 
            'SWAG':{'model_class':SWAG, 'sequential': False }, 
            'LSTM_MC': {'model_class':LSTM_MC, 'sequential': True }, 
            'BNN': {'model_class':BNN, 'sequential': False},
            'NN_Standard': {'model_class':NN_Standard, 'sequential': False },
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
    Nbatches = X_train.shape[0]/args.batch_size

    if args.mode == 'train':
        model.train(train_loader, args.n_epochs, args.batch_size, stats, args.pretrained_dir, Nbatches, args.adversarial_training)

    elif args.mode == 'evaluate':
        if args.model == 'SWAG':
            # BatchNorm buffers update using train dataset.
           results = model.evaluate(test_loader, args.n_samples,  args.pretrained_dir,  train_loader,  args.adversarial_training)

        else:
            results = model.evaluate(test_loader, args.n_samples, args.pretrained_dir, args.adversarial_training)
        
        plot_results(results, args,  stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NN_MC', help='BNN, NN_MC, Deep_Ensemble, LSTM_MC, GNN_MC or SWAG (default: NN_MC)')
    parser.add_argument('--adversarial_training', action='store_true', help='perform adversarial training (default: False)')
    parser.add_argument('--task', type=str, default='regression', help='regression or classification (default: regression)')
    parser.add_argument('--start_train', type=str, default='2019-01-01', help='start date of training (default: 2019-01-01)')
    parser.add_argument('--end_train', type=str, default='2019-12-31', help='end date of training (default: 2019-01-01)')
    parser.add_argument('--start_test', type=str, default='2020-01-01', help='start date of testing (default: 2019-01-01)')
    parser.add_argument('--end_test', type=str, default='2020-02-01', help='end date of testing (default: 2019-01-01)')
    parser.add_argument('--forecast_horizon', type=int, default=24 )
    parser.add_argument('--historical_sequence_length', type=int, default=24)

    parser.add_argument('--mode', type=str, default='evaluate', help='train or evaluate (default: evaluate)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=2000)

    parser.add_argument('--data_dir', type=str, default='./dataset', help='data director (default: ./dataset)')
    parser.add_argument('--plots_dir', type=str, default='./plots',  help='dir for saving results figures (default: ../plots)')
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained',  help='dir for saving trained models (default: ../pretrained)')
    parser.add_argument('--n_samples', type=int, default=1000, help='number of samples to use during inference (default: 1000)')

    args = parser.parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.pretrained_dir, exist_ok=True)
    print('Input Args: ', args)
    run(args)







