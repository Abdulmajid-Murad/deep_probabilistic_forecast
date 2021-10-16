
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from probabilistic_forecast.utils.torch_utils import get_device
from probabilistic_forecast.utils.plot_utils import plot_training_curve, plot_regression, plot_classification


class NN_MC():

    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net_arch = [512, 512]
        self.task = args.task
        self.device = get_device()

        self.network = Network(self.input_dim, self.output_dim, self.net_arch, self.task)
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

        learning_rate=1e-3
        weight_decay = 1e-3
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.999)


        if adversarial_training:
            delta = torch.zeros([batch_size, self.input_dim]).to(self.device)
            X_train_max= torch.tensor(stats['X_train_max'])
            X_train_max = torch.flatten(X_train_max.expand(stats['historical_sequence_length'], -1)).to(self.device)

            X_train_min= torch.tensor(stats['X_train_min'])
            X_train_min = torch.flatten(X_train_min.expand(stats['historical_sequence_length'], -1)).to(self.device)

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
        plot_training_curve(loss_history, lr_history, fig_save_name)
        

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
    def __init__(self, input_dim, output_dim, net_arch, task):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.dropout_probability = 0.5
        self.layers = nn.ModuleList()
        in_features = self.input_dim
        for hidden_size in net_arch:
            self.layers.append(nn.Linear(in_features=in_features, out_features=hidden_size))
            in_features = hidden_size

        if self.task == 'regression':
            self.layers.append(nn.Linear(in_features=in_features, out_features=2*self.output_dim))
            self.Softplus= nn.Softplus()
        elif self.task == 'classification':
            self.layers.append(nn.Linear(in_features=in_features, out_features=self.output_dim))
            self.Sigmoid= nn.Sigmoid()
                                    
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.dropout(x, p=self.dropout_probability, training=True, inplace=True)
            x = self.act(x)
        out = self.layers[-1](x)
 
        if self.task == 'regression':
            mean = out[:, :self.output_dim]
            # The variance should always be positive (softplus) and la
            variance = self.Softplus(out[:, self.output_dim:])+ 1e-06 
            return mean, variance

        elif self.task == 'classification':
            prob = self.Sigmoid(out)
            return prob