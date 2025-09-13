# Block-LISTA: Block Learned Proximal Algorithm

> **⚠️ Disclaimer**: This documentation is AI-generated and may contain errors. Please refer to the actual code for accuracy.
> 
> **🚧 Development Status**: This project is currently under development. The main branch contains numerous bugs and non-standard code practices. Major adjustments and fixes are expected by the end of this year. We recommend waiting for the official release before use.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English | [中文](README_CN.md)

## 📖 Overview

Block-LISTA is a deep learning-based implementation of block learned proximal algorithms, specifically designed for sparse coding and signal reconstruction tasks. This project implements various variants of Learned Iterative Shrinkage-Thresholding Algorithm (LISTA), including traditional ISTA, FISTA, and improved Block-LISTA algorithms.

### Key Features

- 🚀 **Multiple Algorithm Implementations**: ISTA, FISTA, LISTA, ALISTA, TiLISTA, etc.
- 🔧 **Flexible Proximal Operators**: Support for L1, L2,1, L1/2, and other regularizations
- 📊 **Complete Training Pipeline**: Including data generation, model training, validation, and testing
- 🎯 **High-Performance Optimization**: GPU acceleration and batch training support
- 📈 **Experiment Visualization**: Built-in loss visualization and performance metrics calculation

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- ConfigArgParse

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/block-lista.git
cd block-lista

# Install dependencies
pip install torch torchvision numpy matplotlib configargparse

# Or using conda
conda install pytorch torchvision numpy matplotlib
pip install configargparse
```

## 🚀 Quick Start

### Basic Training

```bash
# Train LISTA model with default configuration
python main.py

# Train with custom parameters
python main.py --layers 15 --tau 0.1 --train-size 1024 --init-lr 0.01
```

### Configuration Parameters

Main configuration parameters:

```python
# Problem parameters
--tau 0.1              # Regularization parameter
--m 256                # Number of rows in matrix A
--n 512                # Number of columns in matrix A
--sparsity 8           # Sparsity level
--gLen 16              # Group length

# Model parameters
--layers 10            # Number of network layers

# Training parameters
--train-size 64        # Number of training samples
--val-size 2048        # Number of validation samples
--train-batch-size 128 # Training batch size
--init-lr 0.01         # Initial learning rate
```

### Model Testing

```bash
# Test trained model
python main_test.py --model-path results/temp/LISTA_ProxL2_1over2.pt
```

## 🏗️ Project Structure

```
block-lista/
├── main.py                 # Main training script
├── config.py              # Configuration file
├── loss.py                # Loss function definitions
├── utils.py               # Utility functions
├── models/                # Model definitions
│   ├── __init__.py
│   ├── block_lista.py     # LISTA model
│   ├── block_fista.py     # FISTA model
│   ├── block_ista.py      # ISTA model
│   ├── block_alista.py    # ALISTA model
│   └── ...
├── prox/                  # Proximal operators
│   ├── __init__.py
│   ├── prox_cl.py         # Proximal operator classes
│   └── prox_cl_acc.py     # Accelerated proximal operators
├── common/                # Common modules
│   ├── model.py           # Base model classes
│   └── __init__.py
├── data/                  # Data-related files
└── results/               # Experimental results
```

## 🧠 Model Architecture

### LISTA Algorithm Principle

LISTA (Learned ISTA) is a method that unfolds the traditional Iterative Shrinkage-Thresholding Algorithm (ISTA) into a deep neural network. For the sparse coding problem:

```
min_x (1/2)||Ax - d||²₂ + τ||x||₁
```

The LISTA iterative update formula is:

```
x^(k+1) = S_θ(k)(Bd + Wx^(k))
```

Where:
- `B` and `W` are learnable parameter matrices
- `S_θ` is the soft-thresholding function
- `θ` is the learnable threshold parameter

### Supported Algorithms

| Algorithm | Description | Features |
|-----------|-------------|----------|
| ISTA | Iterative Shrinkage-Thresholding Algorithm | Basic algorithm, slow convergence |
| FISTA | Fast Iterative Shrinkage-Thresholding Algorithm | Accelerated version, faster convergence |
| LISTA | Learned ISTA | End-to-end learning, excellent performance |
| ALISTA | Adaptive LISTA | Adaptive step size adjustment |
| TiLISTA | Time-Invariant LISTA | Parameter sharing, reduces overfitting |

### Proximal Operators

The project supports various proximal operators:

- **L1 Regularization**: `ProxL1` - Standard sparse regularization
- **L2,1 Regularization**: `ProxL2_1` - Group sparse regularization  
- **L1/2 Regularization**: `ProxL1over2` - Non-convex sparse regularization
- **Transformed L1**: `ProxTransformedL1` - Transform domain sparsity

## 📊 Experimental Results

### Performance Comparison

NMSE (Normalized Mean Square Error) comparison on standard sparse coding tasks:

| Algorithm | Layer 1 | Layer 5 | Layer 10 |
|-----------|---------|---------|----------|
| ISTA | -1.34 dB | -1.95 dB | -2.35 dB |
| FISTA | -1.34 dB | -1.95 dB | -2.46 dB |
| LISTA | -3.82 dB | -7.72 dB | -8.85 dB |

### Convergence Analysis

LISTA algorithm has significant advantages over traditional iterative algorithms:

- **Convergence Speed**: 5-10x faster than ISTA/FISTA
- **Reconstruction Accuracy**: 3-6 dB improvement in NMSE
- **Computational Efficiency**: Fixed layers, predictable inference time

## 🔧 API Documentation

### Model Factory

```python
from models import ModelFactory

# Create model
factory = ModelFactory()
model = factory.create(
    'LISTA',
    A=A,                    # Measurement matrix
    tau=0.1,               # Regularization parameter
    layers=10,             # Number of layers
    device='cuda',         # Device
    dtype=torch.float64,   # Data type
    prox_cl=ProxL1()       # Proximal operator
)
```

### Training Interface

```python
# Create training step
train_step = make_train_step(model, optimizer)

# Train one batch
loss = train_step(d_batch, network_layer=current_layer, x_gt=x_batch)
```

### Loss Functions

```python
from loss import objective_val

# Calculate objective function value
loss = objective_val(x_pred, d, x_gt, objective='GT')

# Supported objective function types
# 'OBJECTIVE': LASSO objective function
# 'GT': MSE with ground truth
# 'NMSE': Normalized MSE (dB)
```

## 🎯 Usage Examples

### Custom Proximal Operator

```python
from prox.prox_cl_acc import ProxL2_1over2

# Create L2,1/2 proximal operator
prox_cl = ProxL2_1over2(n=512, gLen=16)

# Use in model
model = ModelFactory().create(
    'LISTA',
    A=A,
    tau=0.1,
    layers=10,
    prox_cl=prox_cl
)
```

### Batch Experiments

```python
# Experiments with different regularization parameters
tau_list = [0.05, 0.1, 0.15, 0.2]
results = []

for tau in tau_list:
    model = create_model(tau=tau)
    loss = train_model(model)
    results.append((tau, loss))
```

## 📈 Visualization

The project provides rich visualization capabilities:

```python
from utils import plot_linechart

# Plot training curves
model_records = [
    ('LISTA', prox_cl, training_losses),
    ('FISTA', prox_cl, fista_losses)
]
plot_linechart(model_records)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. ICML.
2. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. SIAM Journal on Imaging Sciences.
3. Chen, X., Liu, J., Wang, Z., & Yin, W. (2018). Theoretical linear convergence of unfolded ISTA and its practical weights and thresholds. NIPS.

⭐ If this project helps you, please give us a star!