# 下载和缓存数据集

import hashlib
import os
import tarfile
import zipfile

import OpenSSL.crypto
import matplotlib.pyplot as plt
import requests

import numpy as np
import pandas as pd
import torch
import pylab
from torch import nn
from d2l import torch as d2l
from torch.utils import data

# DATA_HUB = dict()
# DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
#
#
#
# def download(name, cache_dir=os.path.join('..', 'data')):
#     '''下载一个DATA_HUB中的文件，返回本地文件名'''
#     assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
#     url, sha1_hash = DATA_HUB[name]
#     os.makedirs(cache_dir, exist_ok=True)
#     fname = os.path.join(cache_dir, url.split('/')[-1])
#     if os.path.exists(fname):
#         sha1 = hashlib.sha1()
#         with open(fname, 'rb') as f:
#             while True:
#                 data = f.read(1048576)
#                 if not data:
#                     break
#                 sha1.update(data)
#         if sha1.hexdigest() == sha1_hash:
#             return fname
#     print(f'正在从{url}下载{fname}...')
#     r = requests.get(url, stream=True, verify=True)
#     with open(fname, 'wb') as f:
#         f.write(r.content)
#     return fname
#
#
# def download_extract(name, folder=None):
#     '''下载并解压zip/tar文件'''
#     fname = download(name)
#     base_dir = os.path.dirname(fname)
#     data_dir, ext = os.path.splitext(fname)
#     if ext == '.zip':
#         fp = zipfile.ZipFile(fname, 'r')
#     elif ext in ('.tar', '.gz'):
#         fp = tarfile.open(fname, 'r')
#     else:
#         assert False, '只有zip/tar文件可以被解压缩'
#     fp.extractall(base_dir)
#     return os.path.join(base_dir, folder) if folder else data_dir
#
#
#
# def download_all():
#     for name in DATA_HUB:
#         download(name)
#
#
# DATA_HUB['kaggle_house_train'] = (
#     DATA_URL + 'kaggle_house_pred_train.csv',
#     '585e9cc93e70b39160e7921475f9bcd7d31219ce')
#
# DATA_HUB['kaggle_house_test'] = (
#     DATA_URL + 'kaggle_house_pred_test.csv',
#     'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


train_data = pd.read_csv('D:\PyCharm\Projects\data\kaggle_house_pred_train.csv')
test_data = pd.read_csv('D:\PyCharm\Projects\data\kaggle_house_pred_test.csv')

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))     # 移除不带有用于预测的信息
print(all_features.shape)

'''数据预处理'''

# 若无法获得训练数据，可根据训练数据计算均值和标准差
'''用于获取all_features中数据类型为数值型（非对象类型）的特征列的索引'''
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
'''对all_features中的数值型特征进行标准化操作。具体地，它将每个数值型特征列的值减去该列的平均值，然后除以该列的标准差。
这个过程将使得每个特征列的值具有零均值和单位方差，从而将特征值范围映射到相对一致的尺度上'''
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
'''将标准化后的数值型特征中的缺失值（NaN）替换为0。这是一种常见的缺失值处理方法，通过将缺失值替换为0，保留了原始数据的分布特征'''
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# dummy_na=True将na(缺失值)视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

print(all_features.shape)


'''将数据转换为张量用于训练'''
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)



'''训练'''
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    # net = nn.Sequential(nn.Linear(in_features, 1))
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(in_features, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1))
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于 1 的值设置为 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


'''Adam 优化器对初始学习率不敏感'''
def train(net, train_features, train_labels, test_features, test_labels, num_epochs,
          learning_rate, weight_deacy, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_deacy)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


'''K折交叉验证'''
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')

            plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 500, 0.1, 3, 256
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')