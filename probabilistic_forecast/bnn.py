
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from probabilistic_forecast.utils.torch_utils import get_device
from probabilistic_forecast.utils.plot_utils import plot_training_curve_bnn, plot_regression, plot_classification


def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term)
    return out


class laplace_prior(object):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b

    def loglike(self, x, do_sum=True):
        if do_sum:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b).sum()
        else:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b)


class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term)
    

class BayesianLayer(nn.Module):

    def __init__(self, input_dim, output_dim, prior_class):
        super().__init__()
        self.n_in = input_dim
        self.n_out = output_dim
        self.prior = prior_class

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

        self.lpw = 0
        self.lqw = 0

    def forward(self, X):

        eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
        eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

        # sample parameters
        std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
        std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

        W = self.W_mu + 1 * std_w * eps_W
        b = self.b_mu + 1 * std_b * eps_b

        output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)

        lqw = isotropic_gauss_loglike(W, self.W_mu, std_w) + isotropic_gauss_loglike(b, self.b_mu, std_b)
        lpw = self.prior.loglike(W) + self.prior.loglike(b)
        return output, lqw-lpw

        
class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, task):
        super().__init__()

#         self.prior_class= laplace_prior(mu=0, b=0.1)
        self.prior_class= isotropic_gauss_prior(mu=0, sigma=0.1)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.task = task
        
        self.layers = nn.ModuleList()
        in_features = self.input_dim
        for hidden_size in self.net_arch:
            self.layers.append(BayesianLayer(input_dim=in_features, output_dim=hidden_size, prior_class=self.prior_class))
            in_features = hidden_size

        if self.task == 'regression':
            self.layers.append(BayesianLayer(input_dim=in_features, output_dim=2*self.output_dim, prior_class=self.prior_class))
            self.Softplus= nn.Softplus()
        elif self.task == 'classification':
            self.layers.append(BayesianLayer(input_dim=in_features, output_dim=self.output_dim, prior_class=self.prior_class))
            self.Sigmoid= nn.Sigmoid()
                                    
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        kl_total = 0
        
        for layer in self.layers[:-1]:
            x, kl = layer(x)
            kl_total += kl 
            x = self.act(x)
        
        out, kl = self.layers[-1](x)
        kl_total += kl

        if self.task == 'regression':
            mean = out[:, :self.output_dim]
            # The variance should always be positive (softplus) and la
            variance = self.Softplus(out[:, self.output_dim:])+ 1e-06 
            return mean, variance, kl_total

        elif self.task == 'classification':
            prob = self.Sigmoid(out)
            return prob, kl_total



class BNN():
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net_arch = [512, 512]
        self.task = args.task
        self.device = get_device()

        self.network = BayesianNetwork(self.input_dim, self.output_dim, self.net_arch, self.task)
        self.network.to(self.device)
        if self.task == "regression":
            self.criterion = torch.nn.GaussianNLLLoss(full=True, reduction='sum')
        elif self.task == "classification":
            self.criterion = nn.BCELoss(reduction='sum')


    def get_loss(self, output, target):
        if self.task == "regression":
            return self.criterion(output[0], target, output[1]), output[2]
        elif self.task == "classification":
            return self.criterion(output[0], target), output[1]


    def train(self, train_loader, n_epochs, batch_size, stats, pre_trained_dir, Nbatches, adversarial_training=True):
        print('Training {} model {} adversarial training. Task: {}'.format(type(self).__name__, 
            'with' if adversarial_training else 'without', self.task))

        learning_rate=1e-3
        weight_decay = 0#1e-3
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.999)


        if adversarial_training:
            delta = torch.zeros([batch_size, self.input_dim]).to(self.device)
            X_train_max= torch.tensor(stats['X_train_max'])
            X_train_max = torch.flatten(X_train_max.expand(stats['historical_sequence_length'], -1)).to(self.device)

            X_train_min= torch.tensor(stats['X_train_min'])
            X_train_min = torch.flatten(X_train_min.expand(stats['historical_sequence_length'], -1)).to(self.device)

            if self.task == 'regression':
                clip_eps = 0.1 / stats['X_train_max'].max()
                fgsm_step = 0.1 / stats['X_train_max'].max()
            elif self.task == 'classification':
                clip_eps = 0.1 / stats['X_train_max'].max()
                fgsm_step = 0.1 / stats['X_train_max'].max()

            n_repeats = 4
            n_epochs = int(n_epochs / n_repeats)
        
        self.network.train()
        n_samples = 3
        loss_history, lr_history, nll_history, kl_history = [], [], [], []
        for epoch in range(1, n_epochs+1 ):
            epoch_loss , epoch_nll, epoch_kl =[], [], []
            for _ , (features , target) in enumerate(train_loader):
                features  = features.to(self.device)
                target = target.to(self.device)
                
                if adversarial_training:
                    for _ in range(n_repeats):
                        delta_batch = delta[0:features.size(0)]
                        delta_batch.requires_grad = True
                        adv_features = features + delta_batch
                        adv_features.clamp_(X_train_min, X_train_max)

                        nll_cum = 0
                        kl_cum = 0 
                        for i in range(n_samples):
                            output = self.network(adv_features)
                            nll_i, kl_i = self.get_loss(output, target)
                            kl_i = kl_i /Nbatches
                            nll_cum += nll_i
                            kl_cum += kl_i
                        nll = nll_cum /n_samples
                        kl = kl_cum / n_samples
                        loss = nll + kl
                        loss.backward()
                        optimizer.step()        
                        optimizer.zero_grad()
                        epoch_loss.append(loss.item())
                        epoch_nll.append(nll.item())
                        epoch_kl.append(kl.item())

                        pert = fgsm_step * torch.sign(delta_batch.grad)
                        delta[0:features.size(0)] += pert.data
                        delta.clamp_(-clip_eps, clip_eps)

                else:
                    nll_cum = 0
                    kl_cum = 0 
                    for i in range(n_samples):
                        output = self.network(features)
                        nll_i, kl_i = self.get_loss(output, target)
                        kl_i = kl_i /Nbatches
                        nll_cum += nll_i
                        kl_cum += kl_i
                    nll = nll_cum /n_samples
                    kl = kl_cum / n_samples
                    loss = nll + kl
                    loss.backward()
                    optimizer.step()        
                    optimizer.zero_grad()
                    epoch_loss.append(loss.item())
                    epoch_nll.append(nll.item())
                    epoch_kl.append(kl.item())

            lr_history.append(optimizer.param_groups[0]['lr'])
            lr_scheduler.step()
            loss_history.append(np.mean(epoch_loss))
            nll_history.append(np.mean(epoch_nll))
            kl_history.append(np.mean(epoch_kl))

            if epoch % 10 == 0:
                print("Epoch: {0:0.3g}, NNL: {1:0.3g}, KL: {2:0.3g},  lr: {3:0.3g}".format(epoch, nll_history[-1],kl_history[-1], lr_history[-1]), end='\r')

        pre_trained_dir = os.path.join(pre_trained_dir, type(self).__name__)
        os.makedirs(pre_trained_dir , exist_ok=True)
        model_save_name = pre_trained_dir + '/trained_network_' + self.task + ('_adv.pt' if adversarial_training else '.pt')
        fig_save_name = pre_trained_dir + '/training_curve_' +self.task + ('_adv.pdf' if adversarial_training else '.pdf')
        torch.save(self.network.state_dict(), model_save_name)
        plot_training_curve_bnn(nll_history, kl_history, lr_history, fig_save_name)
        

    def evaluate(self, test_loader, n_samples, pre_trained_dir, adversarial_training=True):
        print('Evaluating a pretrained {} model {} adversarial training. Task: {}'.format(type(self).__name__, 
            'with' if adversarial_training else 'without', self.task))
        pre_trained_dir = os.path.join(pre_trained_dir, type(self).__name__)
        model_save_name = pre_trained_dir + '/trained_network_'+ self.task + ('_adv.pt' if adversarial_training else '.pt')
        self.network.load_state_dict(torch.load(model_save_name, map_location=self.device))
        self.network.eval()
        if self.task =='regression':
            samples_mean, samples_var = [], []
            for _ in range(n_samples):
                pred_mean_set, pred_var_set = [], []
                for _, (features , _ ) in enumerate(test_loader):
                    features = features.to(self.device)
                    pred_mean, pred_var, _ = self.network(features)
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
                    output, _ = self.network(features)
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


