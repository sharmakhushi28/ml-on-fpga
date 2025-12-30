# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:16:57 2025

@author: khush
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def onehot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = y.reshape(-1, 1)
    return encoder.fit_transform(integer_encoded)

class MyLogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        N, d = X.shape
        self.W = np.zeros((d, y.shape[1]))
        for epoch in range(self.n_iter):
            scores = X @ self.W
            probs = sigmoid(scores)
            error = y - probs
            grad = X.T @ error / N
            self.W += self.lr * grad
        return self

    def predict(self, X):
        scores = X @ self.W
        probs = sigmoid(scores)
        return np.argmax(probs, axis=1)

iris = datasets.load_iris()
X = iris.data
y = iris.target

# First test only on 2 classes (0 and 1)
mask = y < 2
X = X[mask]
y = y[mask]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term
X_train_ = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_test_ = np.column_stack((np.ones(X_test.shape[0]), X_test))

# One-hot encode y
y_train_oh = onehot_encode(y_train)
y_test_oh = onehot_encode(y_test)

# Train Custom Logistic Regression
my_lr = MyLogisticRegression(lr=0.1, n_iter=5000)
my_lr.fit(X_train_, y_train_oh)

y_pred_train = my_lr.predict(X_train_)
y_pred_test = my_lr.predict(X_test_)

print("\n===== My Logistic Regression =====")
print("Weights:\n", my_lr.W)
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test))

# Sklearn Logistic Regression
sk_lr = LogisticRegression(max_iter=5000, solver="lbfgs")
sk_lr.fit(X_train, y_train)

y_pred_train_sk = sk_lr.predict(X_train)
y_pred_test_sk = sk_lr.predict(X_test)

print("\n===== Sklearn Logistic Regression =====")
print("Intercept:", sk_lr.intercept_)
print("Coefficients:\n", sk_lr.coef_)
print("Train Accuracy:", accuracy_score(y_train, y_pred_train_sk))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test_sk))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test_sk))
