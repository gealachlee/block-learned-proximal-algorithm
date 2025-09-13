import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.model import UnrollAlgorithm


__all__ = ['ALISTA']


def calculate_W(A: np.ndarray) -> torch.Tensor:
    """
     determine W without training
    """

    np_A = A
    m, n = np_A.shape
    # 定义 W 矩阵
    W = cp.Variable((m, n))

    # 定义目标函数
    objective = cp.Minimize(cp.norm(W.T @ np_A, 'fro') ** 2)

    # 添加约束条件
    constraints = [W[:, j] @ np_A[:, j] == 1 for j in range(n)]

    # 设置并求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # 输出结果

    print(W.value)
    return torch.tensor(W.value, requires_grad=False)


class ALISTA(UnrollAlgorithm):
    is_trainable = True

    def __init__(self, A, layers, tau, prox_cl, **opts):
        super().__init__(A, layers, tau, **opts)
        self.W = self.calculate_W(A).to(opts.get('device'))
        self.theta = nn.ParameterList()
        self.prox_cl = prox_cl
        self.gamma = nn.ParameterList()
        for i in range(self.layers):
            self.gamma.append(nn.Parameter(torch.tensor(1.0, requires_grad=True)))
            self.theta.append(nn.Parameter(self.tau / self.L_ref))

    def get_optimizer(
            self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        param_groups = []

        # Current layer

        param_groups.append(
            {
                'params': [self.gamma[layer - 1], self.theta[layer - 1]],  # ,
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': [self.gamma[i], self.theta[i]],
                        'lr': init_lr * (lr_decay_layer ** (layer - i - 1))
                    }
                )
            # Stage decay
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups,weight_decay=1e-4)

    def name(self):
        return 'ALISTA'

    def param_groups(self):
        param_groups = []
        for gamma, theta in zip(self.gamma, self.theta):
            param_groups.append({'params': [gamma, theta]})
        return param_groups

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0
        z = x + F.linear(self.gamma[index] * self.W.T, d - (self.A @ x.T).T).T
        Tx = self.prox_cl.prox(z, self.theta[index])
        return Tx

    def forward(self, d, **kwargs):
        K = kwargs.get('K', self.layers)
        xk = d.new_zeros(d.shape[0], self.n)
        for i in range(K):
            xk = self.T(xk, d, index=i)
        return xk

    def calculate_W(self,A):
        try:
            W= np.load('W.npy')
            return torch.tensor(W,requires_grad=False).to('cuda')
        except:
            W=calculate_W(A).to('cuda')
            np.save('W.npy',W.detach().cpu().numpy())
            return W
