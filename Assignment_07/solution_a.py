# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:51:23 2025

@author: khush
"""

import time
import numpy as np
from scipy.special import expit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

n_samples = 2000
random_state = np.random.RandomState(13)

x1 = random_state.uniform(size=n_samples)
x2 = random_state.uniform(size=n_samples)
x3 = random_state.randint(0, 4, size=n_samples)  
x4 = random_state.uniform(size=n_samples)        

X = np.c_[x1, x2, x3, x4 * 10.0].astype(np.float32)

p = expit(np.sin(3 * x1) - 4 * x2 + x3)  
y = random_state.binomial(1, p, size=n_samples).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def report_results(name, y_true_train, y_pred_train, y_true_test, y_pred_test, train_time_s):
    mc_train = int(np.sum(y_true_train != y_pred_train))
    mc_test  = int(np.sum(y_true_test  != y_pred_test))
    acc_train = accuracy_score(y_true_train, y_pred_train)
    acc_test  = accuracy_score(y_true_test,  y_pred_test)
    print(f"\n=== {name} ===")
    print(f"Training time: {train_time_s:.3f} s")
    print(f"Number in train set: {len(y_true_train)}")
    print(f"Misclassified (train): {mc_train:>4d} | Accuracy (train): {acc_train*100:.2f}%")
    print(f"Number in test  set: {len(y_true_test)}")
    print(f"Misclassified (test):  {mc_test:>4d} | Accuracy (test):  {acc_test*100:.2f}%")

t0 = time.time()
lr = LogisticRegression(max_iter=1000, solver="lbfgs")
lr.fit(X_train, y_train)
t1 = time.time()
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
report_results("Logistic Regression", y_train, y_pred_train, y_test, y_pred_test, t1 - t0)

input_dim = X_train.shape[1]
hidden_dim = 32 
torch.manual_seed(13)

class Net(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1) 
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(input_dim, hidden_dim).to(device)


batch_size = 64
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float().unsqueeze(1))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).float().unsqueeze(1))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50 
t0 = time.time()
model.train()
for epoch in range(epochs):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
t1 = time.time()

def predict_with_torch(m, loader):
    m.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = m(xb)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds.append((probs >= 0.5).astype(np.int64))
    return np.concatenate(preds)

y_pred_train = predict_with_torch(model, train_loader)
y_pred_test  = predict_with_torch(model, test_loader)
report_results("PyTorch FCNN (1 hidden layer)", y_train, y_pred_train, y_test, y_pred_test, t1 - t0)

if HAS_LGB:
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=13
    )
    lgbm.fit(X_train, y_train)
    t1 = time.time()
    y_pred_train = lgbm.predict(X_train)
    y_pred_test = lgbm.predict(X_test)
    report_results("LightGBM", y_train, y_pred_train, y_test, y_pred_test, t1 - t0)
else:
    print("\n=== LightGBM ===\nLightGBM not installed in this environment; skipping this model.")

try:
    import pandas as pd
 
except Exception:
    pass
