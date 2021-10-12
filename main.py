import os
import argparse

from utils.data_utils import data_loader
from utils.torch_utils import torch_loader
from nn_mc import NN_MC
from ensemble import Deep_Ensemble
from swag import SWAG
from lstm_mc import LSTM_MC

def run(args):

    if args.model == 'NN_MC':
        model_class = NN_MC
        linear_input = True
    elif args.model == 'Deep_Ensemble':
        model_class = Deep_Ensemble
        linear_input = True
    elif args.model == 'SWAG':
        model_class = SWAG
        linear_input = True
    elif args.model == 'LSTM_MC':
        model_class = LSTM_MC
        linear_input = False


    X_train, y_train, X_test, y_test, stats =data_loader(args.data_dir,
                                                    task=args.task, 
                                                    historical_sequence_length=args.historical_sequence_length, 
                                                    forecast_horizon=args.forecast_horizon, 
                                                    start_train=args.start_train,
                                                    end_train=args.end_train,
                                                    start_test=args.start_test,
                                                    end_test=args.end_test)
    train_loader, test_loader = torch_loader(X_train, y_train, X_test, y_test, args.historical_sequence_length, args.batch_size, linear_input)
    input_dim = X_train.shape[-1] * args.historical_sequence_length if linear_input else X_train.shape[-1] 
    output_dim = y_train.shape[-1]

    model = model_class(input_dim, output_dim, args.task)

    if args.mode == 'train':
        model.train(train_loader, args.n_epochs, args.batch_size, stats, args.pre_trained_dir, args.adversarial_training)

    elif args.mode == 'evaluate':
        if args.model == 'SWAG':
            # BatchNorm buffers update using train dataset.
            model.evaluate(test_loader, args.n_samples, stats, args.pre_trained_dir, args.results_dir, train_loader,  args.adversarial_training)
        else:
            model.evaluate(test_loader, args.n_samples, stats, args.pre_trained_dir, args.results_dir, args.adversarial_training)



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

    parser.add_argument('--data_dir', type=str, default='./data', help='data director (default: ./data)')
    parser.add_argument('--results_dir', type=str, default='./results',  help='dir for saving results figures (default: ../results)')
    parser.add_argument('--pre_trained_dir', type=str, default='./pre_trained',  help='dir for saving trained models (default: ../pre_trained)')
    parser.add_argument('--n_samples', type=int, default=1000, help='number of samples to use during inference (default: 1000)')

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.pre_trained_dir, exist_ok=True)
    print('Input Args: ', args)
    run(args)







