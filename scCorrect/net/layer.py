#!/usr/bin/env python

import math
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function

activation = {
    'relu': nn.ReLU(),
    'rrelu': nn.RReLU(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    '': None
}
class ASRNormBN1d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        '''

        :param dim: C of N,C
        '''
        super(ASRNormBN1d, self).__init__()
        self.eps = eps
        self.num_channels = dim
        self.stan_mid_channel = self.num_channels // 2
        self.rsc_mid_channel = self.num_channels // 16

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.standard_encoder = nn.Linear(dim, self.stan_mid_channel)  # 16
        self.rescale_encoder = nn.Linear(dim, self.rsc_mid_channel)

        # standardization
        self.standard_mean_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.stan_mid_channel, dim)
        )

        self.standard_std_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.stan_mid_channel, dim),
            self.relu
        )

        # Rescaling
        self.rescale_beta_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.rsc_mid_channel, dim),
            self.tanh
        )

        self.rescale_gamma_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.rsc_mid_channel, dim),
            self.sigmoid
        )

        self.lambda_mu = nn.Parameter(torch.empty(1))
        self.lambda_sigma = nn.Parameter(torch.empty(1))

        self.lambda_beta = nn.Parameter(torch.empty(1))
        self.lambda_gamma = nn.Parameter(torch.empty(1))

        self.bias_beta = nn.Parameter(torch.empty(dim))
        self.bias_gamma = nn.Parameter(torch.empty(dim))

        self.drop_out = nn.Dropout(p=0.3)

        # init lambda and bias
        with torch.no_grad():
            init.constant_(self.lambda_mu, self.sigmoid(torch.tensor(-3)))
            init.constant_(self.lambda_sigma, self.sigmoid(torch.tensor(-3)))
            init.constant_(self.lambda_beta, self.sigmoid(torch.tensor(-5)))
            init.constant_(self.lambda_gamma, self.sigmoid(torch.tensor(-5)))
            init.constant_(self.bias_beta, 0.)
            init.constant_(self.bias_gamma, 1.)

    def forward(self, x):
        '''

        :param x: N,C
        :return:
        '''
        N, C = x.size()
        x_mean = torch.mean(x, dim=0)
        x_std = torch.sqrt(torch.var(x, dim=0)) + self.eps

        # standardization
        x_standard_mean = self.standard_mean_decoder(self.standard_encoder(self.drop_out(x_mean.view(1, -1)))).squeeze()
        x_standard_std = self.standard_std_decoder(self.standard_encoder(self.drop_out(x_std.view(1, -1)))).squeeze()

        mean = self.lambda_mu * x_standard_mean + (1 - self.lambda_mu) * x_mean
        std = self.lambda_sigma * x_standard_std + (1 - self.lambda_sigma) * x_std

        mean = mean.reshape((1, C))
        std = std.reshape((1, C))

        x = (x - mean) / std

        # rescaling
        x_rescaling_beta = self.rescale_beta_decoder(self.rescale_encoder(x_mean.view(1, -1))).squeeze()
        x_rescaling_gamma = self.rescale_gamma_decoder(self.rescale_encoder(x_std.view(1, -1))).squeeze()

        beta = self.lambda_beta * x_rescaling_beta + self.bias_beta
        gamma = self.lambda_gamma * x_rescaling_gamma + self.bias_gamma

        beta = beta.reshape((1, C))
        gamma = gamma.reshape((1, C))

        x = x * gamma + beta

        return x

class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class DSBatchNorm2(nn.Module):
    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        super(DSBatchNorm2, self).__init__()
        shape = (1, num_features)
        self.num_features = num_features
        self.n_domain = n_domain
        self.eps = eps
        self.momentum = momentum
        # self.n_domain = n_domain
        # if num_dims == 2:  # 同样是判断是全连层还是卷积层
        #     shape = (1, num_features)
        # else:
        #     shape = (1, num_features, 1, 1)

        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全初始化成0
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))

    def forward(self, x, y):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        x_norm, self.moving_mean, self.moving_var = self.batch_norm(self.training,
                                                                    x, y, self.gamma, self.beta, self.moving_mean,
                                                                    self.moving_var, eps=self.eps, momentum=self.momentum)
        return x_norm

    def batch_norm(self, is_training, x, y, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.1):
        # 判断当前模式是训练模式还是预测模式
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                # out[indices] = self.bns[i](x[indices])
                if not is_training:
                    # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
                    x_hat = (x[indices] - moving_mean) / torch.sqrt(moving_var + eps)
                else:
                    if len(x[indices].shape) == 2:
                        # 使用全连接层的情况，计算特征维上的均值和方差
                        mean = x[indices].mean(dim=0)
                        var = ((x[indices] - mean) ** 2).mean(dim=0)
                    else:
                        # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
                        # x的形状以便后面可以做广播运算
                        mean = x[indices].mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                        var = ((x[indices] - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3,
                                                                                                                  keepdim=True)
                    # 训练模式下用当前的均值和方差做标准化
                    x_hat = (x[indices] - mean) / torch.sqrt(var + eps)
                    # 更新移动平均的均值和方差
                    moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
                    moving_var = momentum * moving_var + (1.0 - momentum) * var

                x_norm = gamma * x_hat + beta  # 拉伸和偏移
                out[indices] = x_norm

            elif len(indices) == 1:
                out[indices] = x[indices]

        return out, moving_mean, moving_var

class DSASRNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([ASRNormBN1d(num_features) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        #                 self.bns[i].training = False
        #                 out[indices] = self.bns[i](x[indices])
        #                 self.bns[i].training = True
        return out

class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        #                 self.bns[i].training = False
        #                 out[indices] = self.bns[i](x[indices])
        #                 self.bns[i].training = True
        return out


class Encoder(nn.Module):
    """
    VAE Encoder
    """

    def __init__(self, input_dim, h_dim):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            encoder configuration, e.g. enc_cfg = [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]
        """
        super().__init__()
        input_dim = input_dim
        input_dim2 = 1024
        h_dim = h_dim
        # encode
        self.fc = nn.Linear(input_dim, input_dim2)
        self.norm = nn.BatchNorm1d(input_dim2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout()

        # reparameterize
        self.mean_fc = nn.Linear(input_dim2, h_dim)
        self.var_fc = nn.Linear(input_dim2, h_dim)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, y=None):
        """
        """
        o = self.fc(x)
        o = self.norm(o)
        o = self.act(o)
        # o = self.dropout(o)
        mean = self.mean_fc(o)
        var = torch.exp(self.var_fc(o))
        z = self.reparameterize(mean, var)
        return z, mean, var


class Decoder(nn.Module):
    """
    VAE Decoder
    """

    def __init__(self, h_dim, input_dim, n_domains=None):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            encoder configuration, e.g. enc_cfg = [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]
        """
        super().__init__()
        input_dim = input_dim
        h_dim = h_dim
        # encode
        self.fc = nn.Linear(h_dim, input_dim)
        # self.norm = DSBatchNorm(input_dim, n_domains)
        self.norm = DSASRNorm(input_dim, n_domains)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, x, y=None):
        """
        """
        o = self.fc(x)
        o = self.norm(o, y)
        o = self.act(o)
        return o
