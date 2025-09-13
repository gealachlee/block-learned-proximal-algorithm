import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

ModelRecord = namedtuple('model_record', ['name', 'prox_cl', 'losses'])


class UnrollAlgorithm(nn.Module):
    is_trainable: bool

    def __init__(self, A, layers, tau, **opts):
        super().__init__()

        self.register_buffer('A', torch.from_numpy(A))
        self.register_buffer('At', torch.from_numpy(np.transpose(A)))

        self.m, self.n = self.A.size()

        self.layers = layers
        self.tau = tau
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.register_buffer('L_ref', self.L_np * torch.ones(1, 1))
        self.register_buffer('gamma', 1 / self.L_ref)

    def forward(self, d, **kwargs):
        raise NotImplementedError
