from config import opts
from models import ModelFactory
from data.block_dataset import create_block_sc_dataset
import torch
import os
import utils
from loss import objective_val
import numpy as np
from common.model import UnrollAlgorithm, ModelRecord
#from prox.prox_cl_acc import *
# from prox.prox_cl_acc import ProxCappedL1over2
#from prox.prox_cl import *
from prox.prox_cl_acc import *
from prox.prox_cl_acc import ProxCappedL1over2

train_seen_loader, val_seen_loader, test_seen_loader, A, b, c = create_block_sc_dataset(opts)

compare_models = [
    # ModelRecord('ISTA',ProxTransformedL1(n=opts.n, gLen=opts.gLen,a=0.5), []),
    # ModelRecord('FISTA', ProxTransformedL1(n=opts.n, gLen=opts.gLen,a=0.5), []),
    # ModelRecord('LISTA', ProxTransformedL1(n=opts.n, gLen=opts.gLen,a=0.5), []),
    # ModelRecord('ISTA',ProxTransformedL1(n=opts.n, gLen=opts.gLen,a=0.5), []),
     ModelRecord('LISTA', ProxCappedL1over2(n=opts.n, gLen=opts.gLen,gamma=1.0), []),
     ModelRecord('LISTA', ProxCappedl1(n=opts.n, gLen=opts.gLen,gamma=1.0), []),
    # ModelRecord('LISTA', ProxTransformedL1(n=opts.n, gLen=opts.gLen,a=0.5), []),
  #  ModelRecord('LISTA',ProxL1_2over3(n=opts.n, gLen=opts.gLen),[])
]

for index, (model_name, prox, losses) in enumerate(compare_models):
    model_: UnrollAlgorithm = \
        ModelFactory().create(
            model_name,
            A=A,
            layers=opts.layers,
            tau=opts.tau,
            prox_cl=prox
        )
    if model_.is_trainable:  # if model is a network
        checkpoint_name = f'{model_.name()}_{model_.prox_cl.name()}.pt'
        # checkpoint_name: str = model_.name() + '.pt'
        save_path = os.path.join(opts.save_dir, checkpoint_name)
        model_.load_state_dict(torch.load(save_path, map_location='cpu'))
    model_ = model_.cuda()

    testing_losses = [0.0]
    opts.logger('Testing losses:')
    for current_layer in range(1, model_.layers + 1):
        with torch.no_grad():
            test_losses = []
            for x_test, d_test in test_seen_loader:
                model_.eval()
                d_test = d_test.to(device=opts.device, dtype=opts.dtype)
                x_test = x_test.to(device=opts.device, dtype=opts.dtype)
                x_pred = model_(d_test, K=current_layer, x_gt=x_test)
                test_loss = objective_val(x_pred, d_test, x_test, 'NMSE').item()
                test_losses.append(test_loss)
            testing_loss = np.mean(test_losses)
            testing_losses.append(testing_loss)
            opts.logger('Layer {} - testing loss: {}'.format(current_layer, testing_loss))
        compare_models[index].losses.append(testing_loss)

utils.plot_linechart(compare_models)
