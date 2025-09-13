
import matplotlib.pyplot as plt
import logging
from typing import List, Dict

import torch
import random

from common.model import ModelRecord
from prox.prox_cl_acc import convert_prox_tex

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b','g']
styles =  ['-', '--', '-.', ':','solid', 'dashed', 'dashdot', 'dotted']
def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr


def plot_linechart(model_record_list:list[ModelRecord]):
    for index, (model_name, prox, losses) in enumerate(model_record_list):
        # color = random.choice(colors)
        # style = random.choice(styles)
        color = colors[index]
        style = styles[index]
        model_name=model_name.replace('LISTA','LPA')
        plt.plot(losses, color=color, linestyle=style, label=f'{model_name} {convert_prox_tex.get(prox.name())}')
        plt.legend()
        plt.ylabel('NMSE (dB)')
        plt.xlabel('Iterations / Layers')
    plt.savefig('save.png',dpi=300)
    plt.show()


def calculate_SPERR(x: torch.Tensor, d: torch.Tensor):
    "FLSNE is the number of “false negative” and SPERR denotes the number of support error."
    N = x.shape[0]
    x_set = set(x.reshape(-1).nonzero().reshape(-1).detach().cpu().tolist())
    d_set = set(d.reshape(-1).nonzero().reshape(-1).detach().cpu().tolist())

    false_positive = len(x_set.difference(d_set)) / N
    false_negative = len(d_set.difference(x_set)) / N
    print(false_negative, false_positive)

    return false_negative, false_positive


def calculate_FLSNE(x: torch.Tensor, d: torch.Tensor):
    "FLSNE is the number of “false negative” and SPERR denotes the number of support error."
    N = x.shape[0]
    x_set = set(x.reshape(-1).nonzero().reshape(-1).detach().cpu().tolist())
    d_set = set(d.reshape(-1).nonzero().reshape(-1).detach().cpu().tolist())
    diff_x = x_set.difference(d_set)
    diff_d = d_set.difference(x_set)

################
"""

Testing losses:
Layer 1 - testing loss:
Testing losses:

Testing losses:

# 
# """
# tmp_results= [[r'FISTA $\ell_{2,1}$', [-1.3329133988854212, -1.8972027536825944, -2.3357643049171943]],
# [r'ISTA $\ell_{1,\frac{1}{2}}$',[ -1.3568243652975283,-1.9733902975429554,-2.375468397843731]],
# [r'FISTA $\ell_{1,\frac{1}{2}}$',[-1.3568243652975283,-1.9733902975429554,-2.4807215367795035]],
# [r'LPA $\ell_{1,\frac{1}{2}}$',[ -3.3535097226230843,-7.787959702527585,-9.730176350823827]],
# [r'LPA $\ell_{2,1}$',[ -3.6312662077311546,-9.361011179089141,-12.936722748779674]]]
# tmp_results=[[r'ISTA $\ell_{1,\frac{1}{2}}$',[-1.339483168749719,-1.9518778563217603, -2.351692204395846]],
#  [r'FISTA $\ell_{1,\frac{1}{2}}$', [-1.339483168749719,-1.9518778563217603, -2.4564083771617664]],
# [r'LPA $\ell_{1,\frac{1}{2}}$',[-3.8157950492400987,-7.717851928210367, -8.853031761380285]]]
# for index, (model_name, losses) in enumerate(tmp_results):
#     # color = random.choice(colors)
#     # style = random.choice(styles)
#     color = colors[index]
#     style = styles[index]
#     model_name = model_name.replace('LISTA', 'LPA')
#     plt.plot(losses, color=color, linestyle=style, label=f'{model_name} ')
#     plt.legend()
#     plt.ylabel('NMSE (dB)')
#     plt.xlabel('Layer')
# plt.show()
# plt.savefig('l1over2.png',dpi=600)
