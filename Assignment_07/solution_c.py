# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 20:28:38 2025

@author: khush
"""

import time
import warnings
import numpy as np
from scipy.special import expit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb
warnings.filterwarnings("ignore")

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

print("\n====================")
print("1) Logistic Regression")
print("====================")

tbeg = time.time()
lr = LogisticRegression(max_iter=1000, solver="lbfgs")
lr.fit(X_train, y_train)
tend = time.time()
print("Total training time ", tend - tbeg)

print('Number in train ', len(y_train))
y_pred = lr.predict(X_train)
mc_train = (y_train != y_pred).sum()
print('Misclassified samples: %d' % mc_train)
acc_train = accuracy_score(y_train, y_pred)
print('Accuracy: %.2f' % acc_train)

print('Number in test ', len(y_test))
y_pred = lr.predict(X_test)
mc_test = (y_test != y_pred).sum()
print('Misclassified samples: %d' % mc_test)
acc_test = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % acc_test)

print("\n====================")
print("2) SVM (Linear Kernel)")
print("====================")

tbeg = time.time()
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
tend = time.time()
print("Total training time ", tend - tbeg)

print('Number in train ', len(y_train))
y_pred = svm_clf.predict(X_train)
mc_train = (y_train != y_pred).sum()
print('Misclassified samples: %d' % mc_train)
acc_train = accuracy_score(y_train, y_pred)
print('Accuracy: %.2f' % acc_train)

print('Number in test ', len(y_test))
y_pred = svm_clf.predict(X_test)
mc_test = (y_test != y_pred).sum()
print('Misclassified samples: %d' % mc_test)
acc_test = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % acc_test)

print("\n====================")
print("3) PyTorch FCNN (1 hidden layer)")
print("====================")

input_dim = X_train.shape[1]
hidden_dim = 32
epochs = 50
batch_size = 64

class Net(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(input_dim, hidden_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_ds = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).float().unsqueeze(1)
)
test_ds = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test).float().unsqueeze(1)
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

tbeg = time.time()
model.train()
for epoch in range(epochs):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
tend = time.time()
print("Training time: %.3f s" % (tend - tbeg))

model.eval()
with torch.no_grad():
    train_logits = model(torch.from_numpy(X_train).float().to(device)).cpu().numpy().ravel()
    test_logits = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy().ravel()

train_probs = 1.0 / (1.0 + np.exp(-train_logits))
test_probs = 1.0 / (1.0 + np.exp(-test_logits))
y_pred_train = (train_probs >= 0.5).astype(int)
y_pred_test = (test_probs >= 0.5).astype(int)

print('Number in train ', len(y_train))
y_pred = y_pred_train
mc_train = (y_train != y_pred).sum()
print('Misclassified samples: %d' % mc_train)
acc_train = accuracy_score(y_train, y_pred)
print('Accuracy: %.2f' % acc_train)

print('Number in test ', len(y_test))
y_pred = y_pred_test
mc_test = (y_test != y_pred).sum()
print('Misclassified samples: %d' % mc_test)
acc_test = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % acc_test)

print("\n====================")
print("4) LightGBM (CPU, lgb.train)")
print("====================")

tbeg = time.time()

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)

params = {
    'force_col_wise': True,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_iterations': 100,
    'n_estimators': 100,
    'max_depth': 5,
    'num_leaves': 100,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 1,
    'lambda_l2': 0.5,
    'random_state': 3
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_test]
)

tend = time.time()
print("Total training time ", tend - tbeg)

print('Number in train ', len(y_train))
y_train_pred = model.predict(X_train)          # probabilities
y_pred = np.where(y_train_pred < 0.5, 0, 1)   # convert to 0/1
mc_train = (y_train != y_pred).sum()
print('Misclassified samples: %d' % mc_train)
acc_train = accuracy_score(y_train, y_pred)
print('Accuracy: %.2f' % acc_train)

print('Number in test ', len(y_test))
y_test_pred = model.predict(X_test)
y_pred = np.where(y_test_pred < 0.5, 0, 1)
mc_test = (y_test != y_pred).sum()
print('Misclassified samples: %d' % mc_test)
acc_test = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % acc_test)
