from mix_trainer.CLISTA import CLISTA
from trainl1spi.train import DeepSimpleFCN
import  torch
import os
import numpy as np
import torch
from config import opts
from data.block_dataset import create_block_sc_dataset,create_mix_sc_dataset
from loss import objective_val
from models import ModelFactory, LISTA

input_dim = 16  # 例如 MNIST 的 28x28=784
hidden_dims = 128  # 4个隐藏层
output_dim = 16  # MNIST 有10个类别

# 初始化模型
learned_l1psi_NN = DeepSimpleFCN(input_dim, hidden_dims, output_dim,num_layers=7)
learned_l1psi_NN.load_state_dict(torch.load(r'D:\keti\block-lista\block-lista\block-lista\trainl1spi\weights.pth'))

# 冻结所有参数
for param in learned_l1psi_NN.parameters():
    if param.count_nonzero()==1:
        param.requires_grad = True
    param.requires_grad = False
learned_l1psi_NN.cuda()



train_seen_loader, val_seen_loader, test_seen_loader, A, b, c =create_mix_sc_dataset(opts)
ATEN=torch.tensor(A).to(device=opts.device, dtype=opts.dtype)

def make_train_step(model, optimizer):
    def train_step(d, network_layer, x_gt=None):
        model.train()  # Set the model to training mode
        x_pred = model(d, K=network_layer, x_gt=x_gt)
        loss = objective_val(x_pred, d, x_gt,A=ATEN)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


model = CLISTA(A,  opts.layers, opts.tau,learned_l1psi_NN)


model = model.to(device=opts.device, dtype=opts.dtype)
assert model.is_trainable is True, 'Model must be trainable'
training_losses = []
validation_losses = []

for j in range(opts.layers):
    current_layer = j + 1

    epoch = 0
    batch_losses = []
    for stage in range(1, 4):
        # Set up optimizer
        optimizer = model.get_optimizer(
            layer=current_layer,
            stage=stage,
            init_lr=opts.init_lr,
            lr_decay_layer=opts.lr_decay_layer,
            lr_decay_stage2=opts.lr_decay_stage2,
            lr_decay_stage3=opts.lr_decay_stage3,
        )
        best_val_nmse = 1e30
        best_val_epoch = epoch
        opts.logger('Training layer {} - stage {}'.format(current_layer, stage))

        train_step = make_train_step(model, optimizer)

        while True:
            for x_batch, d_batch in train_seen_loader:  # Loop over all training batches
                x_batch = x_batch.to(device=opts.device, dtype=opts.dtype)
                d_batch = d_batch.to(device=opts.device, dtype=opts.dtype)
                loss = train_step(d_batch, network_layer=current_layer, x_gt=x_batch)
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)  # Compute the average of the batch losses
            training_losses.append(training_loss)  # Append this new value to the array of losses
            epoch += 1

            with torch.no_grad():
                val_losses = []
                for x_val, d_val in val_seen_loader:  # Loop over all validation batches
                    model.eval()  # Set the model to evaluation mode
                    d_val = d_val.to(device=opts.device, dtype=opts.dtype)
                    x_val = x_val.to(device=opts.device, dtype=opts.dtype)

                    x_pred = model(d_val, K=current_layer, x_gt=x_val)
                    val_loss = objective_val(x_pred, d_val, x_val, 'GT').item()
                    val_losses.append(val_loss)
                validation_loss = np.mean(val_losses)
                validation_losses.append(validation_loss)

                opts.logger(
                    f'[%({epoch})d] Training loss: ({training_loss:.5f})\t Validation loss: ({validation_loss:.5f})'

                )

            if validation_loss < best_val_nmse:
                best_val_nmse = validation_loss
                best_val_epoch = epoch
            if epoch - best_val_epoch > opts.best_wait:
                break

#    checkpoint_name = f'{model.name()}_{model.prox_cl.name()}.pt'
 #   save_path = os.path.join(opts.save_dir, checkpoint_name)
  #  torch.save(model.state_dict(), save_path)
   # opts.logger('Saved the model to file: ' + save_path)
