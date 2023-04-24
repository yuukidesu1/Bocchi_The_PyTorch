# 这是一个对手写数字进行识别的实例，来说明如何借助nn工具箱来实现神经网络（P70）

import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter


train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 20
lr = 0.01
momentum = 0.5

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = mnist.MNIST('../data', train=True, transform=transform, download=False)
test_dataset = mnist.MNIST('../data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 可视化源数据
import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f"Ground Truth:{example_targets[i]}")
    plt.xticks([])
    plt.yticks([])
    # plt.show()

# 构建模型
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()

        # 将输入的张量展平成一维
        self.flatten = nn.Flatten()

        # 第一个全连接层及其批归一化层
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),  # 输入维度in_dim，输出维度n_hidden_1的全连接层
            nn.BatchNorm1d(n_hidden_1)      # 批归一化层，对全连接层输出做归一化处理
        )

        # 第二个全连接层及其批归一化层
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),  # 输入维度n_hidden_1，输出维度n_hidden_2的全连接层
            nn.BatchNorm1d(n_hidden_2)         # 批归一化层，对全连接层输出做归一化处理
        )

        # 输出层，维度为out_dim
        self.out = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)  # 输入维度n_hidden_2，输出维度out_dim的全连接层
        )

    def forward(self, x):
        # 展平输入张量
        x = self.flatten(x)

        # 通过第一个全连接层及其批归一化层，使用ReLU激活函数
        x = F.relu(self.layer1(x))

        # 通过第二个全连接层及其批归一化层，使用ReLU激活函数
        x = F.relu(self.layer2(x))

        # 输出层使用softmax激活函数，返回结果
        x = F.softmax(self.out(x), dim=1)
        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(28*28, 300, 100, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# 训练模型
losses = []
acces = []
eval_losses = []
eval_acces = []
writer = SummaryWriter(log_dir='logs', comment='train-loss')

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    # 修改动态学习率
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.9
        print("学习率：{:.6f}".format(optimizer.param_groups[0]['lr']))
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        # 正向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 保存loss的数据与epoch数值
        writer.add_scalar('Train', train_loss/len(train_loader), epoch)
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # 在测试集上校验结果
    eval_loss = 0
    eval_acc = 0
    # net.eval() # 将模型改为预测模式
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch：{:.4f}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), eval_loss / len(test_loader),
                  eval_acc / len(test_loader)))