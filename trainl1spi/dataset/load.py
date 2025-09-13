import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
import pickle

x = np.load(r'D:\keti\block-lista\block-lista\block-lista\trainl1spi\dataset\x.npy')
y= np.load(r'D:\keti\block-lista\block-lista\block-lista\trainl1spi\dataset\y.npy')

50000*0.8
train_size = int(0.8 * len(x))
val_size = int(0.1 * len(x))
test_size = len(x) - train_size - val_size

def load():
    # Transform x and d into tensors
    x_tensor = torch.from_numpy(x).float()
    d_tensor = torch.from_numpy(y).float()


    seen_dataset = TensorDataset(x_tensor, d_tensor)
    train_seen_dataset, val_seen_dataset, test_seen_dataset = random_split(
        seen_dataset, [train_size, val_size, test_size])

    # Create loaders for batches from training and validation datasets
    train_seen_loader = DataLoader(dataset=train_seen_dataset,
                                   batch_size=128,
                                   shuffle=True)
    val_seen_loader = DataLoader(dataset=val_seen_dataset,
                                 batch_size=5000,
                                 shuffle=False)
    test_seen_loader = DataLoader(dataset=test_seen_dataset,
                              batch_size=5000,
                              shuffle=False)
    return train_seen_loader, val_seen_loader, test_seen_loader, x, y


