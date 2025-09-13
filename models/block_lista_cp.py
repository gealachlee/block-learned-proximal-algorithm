import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.model import UnrollAlgorithm
from prox import *

__all__ = ['LISTA_CP']


class LISTA_CP(UnrollAlgorithm):
    is_trainable = True

    def __init__(self, A, layers, tau, prox_cl, **opts):
        super().__init__(A, layers, tau, **opts)
        self.W = nn.ParameterList()
        self.theta =nn.ParameterList()
      #  self.theta2= nn.Parameter(self.tau / self.L_ref)
        self.prox_cl = prox_cl
        for i in range(self.layers):
            self.theta.append(self.tau / self.L_ref)
            self.W.append(self.A / self.L_ref)

    def get_optimizer(
            self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        param_groups = []

        # Current layer
        param_groups.append(
            {
                'params': [self.W[layer - 1]],  # ,
                'lr': init_lr
            }
        )
        param_groups.append(
            {
                'params': [self.theta[layer-1]],  # ,
                'lr': init_lr
            }
        )


       # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': [self.W[i]],  #
                        'lr': init_lr * (lr_decay_layer ** (layer - i - 1))
                    }
                )
                param_groups.append(
                    {
                        'params': [self.theta[i]],
                        'lr': init_lr * (lr_decay_layer ** (layer - i - 1))
                    }
                )
            # Stage decay
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups,weight_decay=1e-3)

    def name(self):
        return 'LISTA_CP'

    def param_groups(self):
        param_groups = []
        for gamma, theta in zip(self.gamma, self.theta):
            param_groups.append({'params': [gamma, theta]})
        return param_groups

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0
        z = x +F.linear(self.W[index].T, d - (self.A @ x.T).T).T

        Tx = self.prox_cl.prox(z, self.theta[index])
       # Tx = self.prox_cl.prox(z,0.5*torch.abs(self.theta[index-1])+0.5*torch.abs(self.theta[index]))
        return Tx

    def forward(self, d, **kwargs):
        K = kwargs.get('K', self.layers)
        xk = d.new_zeros(d.shape[0], self.n)
        for i in range(K):
            xk = self.T(xk, d, index=i)
        return xk
