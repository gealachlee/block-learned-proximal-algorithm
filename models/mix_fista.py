import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import prox
from common.model import UnrollAlgorithm
from prox.prox_cl_acc import shrink
__all__ = ['MIX_FISTA']


class MIX_FISTA(UnrollAlgorithm):
    is_trainable = False

    def __init__(self, A, layers, tau, prox_cl1, **opts):
        super().__init__(A, layers, tau, **opts)
        self.prox_cl1 = prox_cl1

    def name(self):
        return 'MIX_FISTA'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0
        r = F.linear(x, self.A) - d.cuda()
        z = x - self.gamma * F.linear(r, self.At)

        Tx = self.prox_cl1.prox(z,  self.gamma * tau)  # new
        Tx = shrink(Tx, self.gamma *tau)
        return Tx

    def forward(self, d, **kwargs):
        K = kwargs.get('K', self.layers)
        with torch.no_grad():
            xk = d.new_zeros(d.shape[0], self.n)
            zk = d.new_zeros(d.shape[0], self.n)
            tk = 1.0
            for i in range(K):
                x_next = self.T(zk, d, index=i)

                t_next = 0.5 + np.sqrt(1.0 + 4.0 * tk ** 2) / 2.0
                z_next = x_next + (tk - 1.0) / t_next * (x_next - xk)

                xk = x_next
                zk = z_next
                tk = t_next

        return xk
