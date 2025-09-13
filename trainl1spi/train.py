
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from trainl1spi.dataset.load import load

from prox.prox_cl_acc import shrink

class DeepSimpleFCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout_rate=0.2):
        """
        参数说明：
            input_dim (int): 输入特征维度（如 784）
            hidden_dim (int): 所有隐藏层的统一维度（如 256）
            output_dim (int): 输出维度（如 10）
            num_layers (int): 总层数（输入层 + 隐藏层 + 输出层，默认8层）
            dropout_rate (float): Dropout 概率（默认0.3）
        """
        super(DeepSimpleFCN, self).__init__()
        self.num_layers = num_layers
        # 输入层
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.eta= nn.Parameter(torch.ones(1,1)*0.001)
        # 隐藏层（num_layers-2 层，因为输入和输出各占1层）
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for index,layer in enumerate(self.layers[:-1]):  # 除输出层外所有层
            #x = F.leaky_relu(layer(x),negative_slope=0.01)
            if index<5:
                x = F.leaky_relu(layer(x),negative_slope=0.01)
                #x = self.dropout(x)
            else:
                x = shrink(layer(x), (self.num_layers-index)*0.001)
           #
        x = self.layers[-1](x)  # 输出层（无激活函数）
        x = shrink(x, self.eta)
        return x

import torch.optim as optim
# 定义损失函数和优化器

# 示例用法
if __name__ == "__main__":
    # 定义网络参数
    input_dim = 16  # 例如 MNIST 的 28x28=784
    hidden_dims = 128 # 4个隐藏层
    output_dim = 16  # MNIST 有10个类别

    # 初始化模型
    model = DeepSimpleFCN(input_dim, hidden_dims, output_dim,num_layers=7)
    print(model)

    # 模拟输入数据（batch_size=32）
   # dummy_input = torch.randn(32, input_dim)
    train_seen_loader, val_seen_loader, test_seen_loader, x, y = load()

    criterion = nn.SmoothL1Loss()  # 分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

    epoch=0
    while True:
        for x_batch, d_batch in train_seen_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, d_batch)  # 模拟标签
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        epoch += 1
        with torch.no_grad():
            for x_batch2, d_batch2 in val_seen_loader:
                outputs = model(x_batch2)
              #  print(d_batch2)
                loss = criterion(outputs, d_batch2)  # 模拟标签
                print(f"Epoch {epoch}, val Loss: {loss.item():.6f}\n")
        if epoch > 25:
            break

    with torch.no_grad():
        for x_batch3, d_batch3 in test_seen_loader:
            outputs = model(x_batch3)
            print(d_batch3)
            loss = criterion(outputs, d_batch3)  # 模拟标签
            print(f"Epoch {epoch}, val Loss: {loss.item():.6f}\n")


    #
    model
#torch.save(model.state_dict(), 'weights.pth')