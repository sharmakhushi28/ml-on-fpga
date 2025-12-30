# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 20:48:43 2025

@author: khush
"""

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

n_samples = 2000
random_state = np.random.RandomState(13)

x1 = random_state.uniform(size=n_samples)
x2 = random_state.uniform(size=n_samples)
x3 = random_state.randint(0, 4, size=n_samples)
x4 = random_state.uniform(size=n_samples)

X = np.c_[x1, x2, x3, x4 * 10].astype(np.float32)
p = expit(np.sin(3 * x1) - 4 * x2 + x3)
y = random_state.binomial(1, p, size=n_samples).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

XX_train = torch.from_numpy(X_train).float()
XX_test  = torch.from_numpy(X_test).float()

targets0 = np.eye(2)[y_train.astype(int)]
yy_train = torch.from_numpy(targets0).float()

targets1 = np.eye(2)[y_test.astype(int)]
yy_test  = torch.from_numpy(targets1).float()

input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 2

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
tbeg = time.time()

for epoch in range(epochs):
    inputs = Variable(XX_train)
    labels = Variable(yy_train)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, torch.max(labels, 1)[1])
    loss.backward()
    optimizer.step()

tend = time.time()
print("Training time:", tend - tbeg)

train_logits = model(XX_train)
_, predicted_train = torch.max(train_logits.data, 1)
y_pred_train = predicted_train.numpy()

print('Number in train ', len(y_train))
mc_train = (y_train != y_pred_train).sum()
print('Misclassified samples: %d' % mc_train)

from sklearn.metrics import accuracy_score
acc_train = accuracy_score(y_train, y_pred_train)
print('Accuracy: %.2f' % acc_train)

test_logits = model(XX_test)
_, predicted_test = torch.max(test_logits.data, 1)
y_pred_test = predicted_test.numpy()

print('Number in test ', len(y_test))
mc_test = (y_test != y_pred_test).sum()
print('Misclassified samples: %d' % mc_test)
acc_test = accuracy_score(y_test, y_pred_test)
print('Accuracy: %.2f' % acc_test)
