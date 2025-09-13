import math
import torch
import torch.nn.functional as F
from common.model import UnrollAlgorithm
import prox

__all__ = ['ISTA']


class ISTA(UnrollAlgorithm):
    is_trainable = False

    def __init__(self, A, layers, tau,prox_cl, **opts):
        super().__init__(A, layers, tau, **opts)
        self.prox_cl = prox_cl

    def name(self):
        return 'ISTA'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0
        r = F.linear(x, self.A) - d.cuda()
        z = x - self.gamma * F.linear(r, self.At)
        Tx = self.prox_cl.prox(z, self.gamma * tau)
        return Tx

    def forward(self, d, **kwargs):
        K = kwargs.get('K', self.layers)
        with torch.no_grad():
            xk = d.new_zeros(d.shape[0], self.n)
            for i in range(K):
                xk = self.T(xk, d, index=i)
        return xk
