# Block-LISTA: 块学习近端算法

> **⚠️ 声明**: 本文档由AI生成，可能存在输出错误的地方，请以实际代码为准。
> 
> **🚧 开发状态**: 项目目前还在开发中，主分支上的代码有较多漏洞和代码不规范的地方，预计今年年底会有重大调整和修复。建议等待正式版本发布后使用。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | 中文

## 📖 项目简介

Block-LISTA 是一个基于深度学习的块学习近端算法实现，专门用于稀疏编码和信号重建任务。该项目实现了多种学习型迭代收缩阈值算法（LISTA）的变体，包括传统的 ISTA、FISTA 以及改进的 Block-LISTA 算法。

### 核心特性

- 🚀 **多种算法实现**: ISTA, FISTA, LISTA, ALISTA, TiLISTA 等
- 🔧 **灵活的近端算子**: 支持 L1, L2,1, L1/2 等多种正则化
- 📊 **完整的训练流程**: 包含数据生成、模型训练、验证和测试
- 🎯 **高性能优化**: 支持 GPU 加速和批处理训练
- 📈 **实验结果可视化**: 内置损失函数可视化和性能指标计算

## 🛠️ 安装配置

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- ConfigArgParse

### 快速安装

```bash
# 克隆项目
git clone https://github.com/your-username/block-lista.git
cd block-lista

# 安装依赖
pip install torch torchvision numpy matplotlib configargparse

# 或使用 conda
conda install pytorch torchvision numpy matplotlib
pip install configargparse
```

## 🚀 快速开始

### 基本训练

```bash
# 使用默认配置训练 LISTA 模型
python main.py

# 自定义参数训练
python main.py --layers 15 --tau 0.1 --train-size 1024 --init-lr 0.01
```

### 配置参数

主要配置参数说明：

```python
# 问题参数
--tau 0.1              # 正则化参数
--m 256                # 矩阵 A 的行数
--n 512                # 矩阵 A 的列数
--sparsity 8           # 稀疏度
--gLen 16              # 组长度

# 模型参数
--layers 10            # 网络层数

# 训练参数
--train-size 64        # 训练样本数
--val-size 2048        # 验证样本数
--train-batch-size 128 # 训练批大小
--init-lr 0.01         # 初始学习率
```

### 测试模型

```bash
# 测试训练好的模型
python main_test.py --model-path results/temp/LISTA_ProxL2_1over2.pt
```

## 🏗️ 项目结构

```
block-lista/
├── main.py                 # 主训练脚本
├── config.py              # 配置文件
├── loss.py                # 损失函数定义
├── utils.py               # 工具函数
├── models/                # 模型定义
│   ├── __init__.py
│   ├── block_lista.py     # LISTA 模型
│   ├── block_fista.py     # FISTA 模型
│   ├── block_ista.py      # ISTA 模型
│   ├── block_alista.py    # ALISTA 模型
│   └── ...
├── prox/                  # 近端算子
│   ├── __init__.py
│   ├── prox_cl.py         # 近端算子类
│   └── prox_cl_acc.py     # 加速近端算子
├── common/                # 通用模块
│   ├── model.py           # 基础模型类
│   └── __init__.py
├── data/                  # 数据相关
└── results/               # 实验结果
```

## 🧠 模型架构

### LISTA 算法原理

LISTA（Learned ISTA）是一种将传统的迭代收缩阈值算法（ISTA）展开为深度神经网络的方法。对于稀疏编码问题：

```
min_x (1/2)||Ax - d||²₂ + τ||x||₁
```

LISTA 的迭代更新公式为：

```
x^(k+1) = S_θ(k)(Bd + Wx^(k))
```

其中：
- `B` 和 `W` 是可学习的参数矩阵
- `S_θ` 是软阈值函数
- `θ` 是可学习的阈值参数

### 支持的算法

| 算法 | 描述 | 特点 |
|------|------|------|
| ISTA | 迭代收缩阈值算法 | 基础算法，收敛慢 |
| FISTA | 快速迭代收缩阈值算法 | 加速版本，收敛更快 |
| LISTA | 学习型 ISTA | 端到端学习，性能优异 |
| ALISTA | 自适应 LISTA | 自适应步长调整 |
| TiLISTA | 时间不变 LISTA | 参数共享，减少过拟合 |

### 近端算子

项目支持多种近端算子：

- **L1 正则化**: `ProxL1` - 标准稀疏正则化
- **L2,1 正则化**: `ProxL2_1` - 组稀疏正则化  
- **L1/2 正则化**: `ProxL1over2` - 非凸稀疏正则化
- **变换 L1**: `ProxTransformedL1` - 变换域稀疏

## 📊 实验结果

### 性能对比

在标准稀疏编码任务上的 NMSE（归一化均方误差）对比：

| 算法 | Layer 1 | Layer 5 | Layer 10 |
|------|---------|---------|----------|
| ISTA | -1.34 dB | -1.95 dB | -2.35 dB |
| FISTA | -1.34 dB | -1.95 dB | -2.46 dB |
| LISTA | -3.82 dB | -7.72 dB | -8.85 dB |

### 收敛性分析

LISTA 算法相比传统迭代算法具有显著优势：

- **收敛速度**: 比 ISTA/FISTA 快 5-10 倍
- **重建精度**: NMSE 提升 3-6 dB
- **计算效率**: 固定层数，推理时间可预测

## 🔧 API 文档

### 模型工厂

```python
from models import ModelFactory

# 创建模型
factory = ModelFactory()
model = factory.create(
    'LISTA',
    A=A,                    # 测量矩阵
    tau=0.1,               # 正则化参数
    layers=10,             # 层数
    device='cuda',         # 设备
    dtype=torch.float64,   # 数据类型
    prox_cl=ProxL1()       # 近端算子
)
```

### 训练接口

```python
# 创建训练步骤
train_step = make_train_step(model, optimizer)

# 训练一个批次
loss = train_step(d_batch, network_layer=current_layer, x_gt=x_batch)
```

### 损失函数

```python
from loss import objective_val

# 计算目标函数值
loss = objective_val(x_pred, d, x_gt, objective='GT')

# 支持的目标函数类型
# 'OBJECTIVE': LASSO 目标函数
# 'GT': 与真值的 MSE
# 'NMSE': 归一化 MSE (dB)
```

## 🎯 使用示例

### 自定义近端算子

```python
from prox.prox_cl_acc import ProxL2_1over2

# 创建 L2,1/2 近端算子
prox_cl = ProxL2_1over2(n=512, gLen=16)

# 在模型中使用
model = ModelFactory().create(
    'LISTA',
    A=A,
    tau=0.1,
    layers=10,
    prox_cl=prox_cl
)
```

### 批量实验

```python
# 不同正则化参数的实验
tau_list = [0.05, 0.1, 0.15, 0.2]
results = []

for tau in tau_list:
    model = create_model(tau=tau)
    loss = train_model(model)
    results.append((tau, loss))
```

## 📈 可视化

项目提供了丰富的可视化功能：

```python
from utils import plot_linechart

# 绘制训练曲线
model_records = [
    ('LISTA', prox_cl, training_losses),
    ('FISTA', prox_cl, fista_losses)
]
plot_linechart(model_records)
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📚 参考文献

1. Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. ICML.
2. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. SIAM Journal on Imaging Sciences.
3. Chen, X., Liu, J., Wang, Z., & Yin, W. (2018). Theoretical linear convergence of unfolded ISTA and its practical weights and thresholds. NIPS.



⭐ 如果这个项目对您有帮助，请给我们一个星标！