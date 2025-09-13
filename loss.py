import numpy as np
from typing import Union
import torch
from config import opts
from torch.nn import functional as F


def objective_val(x, d, x_gt=None, objective=None, **kwargs):
    if objective is None:
        objective = opts.objective

    # ----------------------------------------------------------
    # Option 1: f(x) = (1/2)*|A*x-d|_2^2 + tau * |x|_1
    #           i.e. the LASSO objective function
    # ----------------------------------------------------------
    if objective == 'OBJECTIVE':
        residual = ((F.linear(x, kwargs.get('A')) - d) ** 2.0).sum(dim=1).mean()
        l1_norm = x.abs().sum(dim=1).mean()
        val = residual / 2.0   + opts.tau * l1_norm

    elif objective == 'GT':
        val = ((x - x_gt) ** 2).sum(dim=1).mean()
    elif objective == 'GT2':
        l1_norm = x.abs().sum(dim=1).mean()
        val = (((x - x_gt) ** 2).sum(dim=1).mean() + opts.tau * l1_norm)

    elif objective == 'NMSE':
        l2 = ((x - x_gt) ** 2).mean()
        denom = (x_gt ** 2).mean()
        val = 10.0 * torch.log10(l2 / denom)

    else:
        raise ValueError('Invalid objective option {}'.format(opts.objective))

    return val
