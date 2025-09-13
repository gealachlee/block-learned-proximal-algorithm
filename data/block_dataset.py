import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torch
from sklearn.preprocessing import normalize


def create_block_sc_dataset(opts):
    m = opts.m
    n = opts.n
    sparsity = opts.sparsity
    gLen = opts.gLen
    gNo = int(opts.n / gLen)

    seen_data_size = opts.train_size + opts.val_size + opts.test_size

    np.random.seed(opts.data_seed)
    torch.random.manual_seed(opts.data_seed)

    A = np.random.normal(size=(m, n))
    A = normalize(A, norm='l2', axis=0)

    gNo1 = np.arange(1, gNo + 1)
    c = np.random.normal(0, 1, (seen_data_size, n, 1))

    for sample in range(c.shape[0]):
        Bs = np.zeros((n, 1))
        ActInd = np.random.choice(gNo1, sparsity, replace=False)
        for i in range(sparsity):
            Bs[((ActInd[i] - 1) * gLen):(ActInd[i] * gLen)] = np.ones((gLen, 1))
        c[sample] = Bs * c[sample]

    if opts.snr and opts.snr < 100000:
        std = np.std(c, axis=1) * np.power(10.0, -opts.snr / 20.0)
        noise = np.random.normal(size=(c.shape[0],c.shape[1]), scale=std).astype(c.dtype)
        print('noise added')
    else:
        noise = 0
    b = A.dot(c) + noise

    # Transform x and d into tensors
    x_tensor = torch.from_numpy(c)[:, :, 0].float()

    d_tensor = torch.from_numpy(b)[:, :, 0].float().T

    seen_dataset = TensorDataset(x_tensor, d_tensor)

    # Split the seen dataset into training, validation and testing portions
    train_seen_dataset, val_seen_dataset, test_seen_dataset = random_split(
        seen_dataset, [opts.train_size, opts.val_size, opts.test_size])

    # Create loaders for batches from training and validation datasets
    train_seen_loader = DataLoader(dataset=train_seen_dataset,
                                   batch_size=opts.train_batch_size,
                                   shuffle=True)
    val_seen_loader = DataLoader(dataset=val_seen_dataset,
                                 batch_size=opts.val_batch_size,
                                 shuffle=False)
    test_seen_loader = DataLoader(dataset=test_seen_dataset,
                                  batch_size=opts.test_batch_size,
                                  shuffle=False)
    return train_seen_loader, val_seen_loader, test_seen_loader, A, b, c


def create_mix_sc_dataset(opts):
    m = opts.m
    n = opts.n
    sparsity = opts.sparsity
    gLen = opts.gLen
    gNo = int(opts.n / gLen)

    seen_data_size = opts.train_size + opts.val_size + opts.test_size

    np.random.seed(opts.data_seed)
    torch.random.manual_seed(opts.data_seed)

    A = np.random.normal(size=(m, n))
    A = normalize(A, norm='l2', axis=0)

    gNo1 = np.arange(1, gNo + 1)
    p = np.random.uniform(low=0.3, high=0.3, size=[seen_data_size, opts.n])
    bernoulli_terms = np.random.binomial(size=[seen_data_size, opts.n], n=1, p=p)
    gaussian_terms = np.random.normal(size=[seen_data_size, opts.n], loc=opts.x_mu, scale=opts.x_sigma)
    c = np.multiply(bernoulli_terms, gaussian_terms)
    c=c.reshape(c.shape[0],c.shape[1],1)
    for sample in range(c.shape[0]):
        Bs = np.zeros((n, 1))
        ActInd = np.random.choice(gNo1, sparsity, replace=False)
        for i in range(sparsity):
            Bs[((ActInd[i] - 1) * gLen):(ActInd[i] * gLen)] = np.ones((gLen, 1))
        c[sample] = Bs * c[sample]


    if opts.snr and opts.snr < 100000:
        sigma = 1e-3
        #noise = sigma * np.random.random(0, 1, (m, 1))
        noise = sigma*np.random.normal(size=(c.shape[0], c.shape[1])).astype(c.dtype)
        print('noise added')
    else:
        noise = 0
    b = A.dot(c) + noise

    # Transform x and d into tensors
    x_tensor = torch.from_numpy(c)[:, :, 0].float()

    d_tensor = torch.from_numpy(b)[:, :, 0].float().T

    seen_dataset = TensorDataset(x_tensor, d_tensor)

    # Split the seen dataset into training, validation and testing portions
    train_seen_dataset, val_seen_dataset, test_seen_dataset = random_split(
        seen_dataset, [opts.train_size, opts.val_size, opts.test_size])

    # Create loaders for batches from training and validation datasets
    train_seen_loader = DataLoader(dataset=train_seen_dataset,
                                   batch_size=opts.train_batch_size,
                                   shuffle=True)
    val_seen_loader = DataLoader(dataset=val_seen_dataset,
                                 batch_size=opts.val_batch_size,
                                 shuffle=False)
    test_seen_loader = DataLoader(dataset=test_seen_dataset,
                                  batch_size=opts.test_batch_size,
                                  shuffle=False)
    return train_seen_loader, val_seen_loader, test_seen_loader, A, b, c