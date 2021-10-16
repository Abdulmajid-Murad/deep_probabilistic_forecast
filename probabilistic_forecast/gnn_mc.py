
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from probabilistic_forecast.utils.torch_utils import get_device
from probabilistic_forecast.utils.plot_utils import plot_training_curve, plot_regression, plot_classification
from probabilistic_forecast.utils.gnn_utils import GLU, StockBlockLayer



class GNN_MC():

    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.task = args.task
        self.window_size= args.historical_sequence_length
        self.device = get_device()
        self.multi_layer=3
        self.network=Network(units=self.input_dim, stack_cnt=2, time_step=self.window_size, multi_layer=self.multi_layer, output_length=self.output_dim , task=self.task)

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

        learning_rate=1e-4
        weight_decay = 1e-3
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        lr_scheduler = ExponentialLR(optimizer, gamma=0.999)

        if adversarial_training:

            delta = torch.zeros([batch_size, stats['historical_sequence_length'], self.input_dim]).to(self.device)
            X_train_max= torch.tensor(stats['X_train_max'])
            X_train_max= X_train_max.to(self.device)

            X_train_min= torch.tensor(stats['X_train_min'])
            X_train_min= X_train_min.to(self.device)

            if self.task == 'regression':
                clip_eps = 5.0 / stats['X_train_max'].max()
                fgsm_step = 5.0 / stats['X_train_max'].max()
            elif self.task == 'classification':
                clip_eps = 1.0 / stats['X_train_max'].max()
                fgsm_step = 1.0 / stats['X_train_max'].max()

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
   
    def apply_dropout(self, module):
        if type(module) == nn.Dropout:
            module.train()
        
    def evaluate(self, test_loader, n_samples, pre_trained_dir, adversarial_training=True):
        print('Evaluating a pretrained {} model {} adversarial training. Task: {}'.format(type(self).__name__, 
            'with' if adversarial_training else 'without', self.task))
        pre_trained_dir = os.path.join(pre_trained_dir, type(self).__name__)
        model_save_name = pre_trained_dir + '/trained_network_'+ self.task + ('_adv.pt' if adversarial_training else '.pt')
        self.network.load_state_dict(torch.load(model_save_name))
        self.network.eval()
        self.network.apply(self.apply_dropout)

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
    def __init__(self, units, stack_cnt, time_step, multi_layer,output_length, task,  horizon=1, dropout_rate=0.5, leaky_rate=0.2):
        super().__init__()
        self.output_length = output_length
        self.unit = units
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_probability = dropout_rate
        self.task = task

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit) #, dropout=self.dropout_probability)
        self.GRU = nn.GRU(self.time_step, self.unit) #, dropout=self.dropout_probability)

        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i, dropout_rate=self.dropout_probability) for i in range(self.stack_cnt)])

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.task == 'regression':
            self.fc = nn.Sequential(
            nn.Linear(int(self.time_step*self.unit), int(self.time_step*self.unit)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(int(self.time_step*self.unit), 2*self.output_length),
            )
            self.Softplus= nn.Softplus()

        elif self.task == 'classification':
            self.fc = nn.Sequential(
            nn.Linear(int(self.time_step*self.unit), int(self.time_step*self.unit)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(int(self.time_step*self.unit), self.output_length),
            )
            self.Sigmoid= nn.Sigmoid()
  

    def get_laplacian(self, graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = forecast.view(forecast.shape[0], -1)
        forecast = self.fc(forecast)

        if self.task == 'regression':
            mean = forecast[:, :self.output_length]
            variance = self.Softplus(forecast[:, self.output_length:])+ 1e-06
            return mean, variance

        elif self.task == 'classification':
            prob = self.Sigmoid(forecast)
            return prob