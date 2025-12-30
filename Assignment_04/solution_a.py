# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:01:17 2025

@author: khush
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def onehot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = y.reshape(-1, 1)
    return encoder.fit_transform(integer_encoded)

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000, batch_size=1, multiclass=False):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.multiclass = multiclass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.multiclass:
            n_classes = y.shape[1]
            self.W = np.zeros((n_features, n_classes))
        else:
            self.W = np.zeros((n_features, 1))
        
        errors = []
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                if self.multiclass:
                    z = X_batch.dot(self.W)
                    y_pred = softmax(z)
                    grad = X_batch.T.dot(y_pred - y_batch) / len(y_batch)
                else:
                    z = X_batch.dot(self.W)
                    y_pred = sigmoid(z)
                    grad = X_batch.T.dot(y_pred - y_batch) / len(y_batch)
                
                self.W -= self.lr * grad

          
            if self.multiclass:
                probs = softmax(X.dot(self.W))
                loss = -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
            else:
                probs = sigmoid(X.dot(self.W))
                loss = -np.mean(y*np.log(probs+1e-9)+(1-y)*np.log(1-probs+1e-9))
            errors.append(loss)
        
        self.errors = errors

    def predict(self, X):
        if self.multiclass:
            probs = softmax(X.dot(self.W))
            return np.argmax(probs, axis=1)
        else:
            probs = sigmoid(X.dot(self.W))
            return (probs >= 0.5).astype(int).flatten()


iris = datasets.load_iris()
X = iris.data[:, 0:4]
y = iris.target

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add intercept
X = np.column_stack((np.ones(X.shape[0]), X))

# Binary classification 
X_bin = X[0:100]
y_bin = y[0:100]

X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

y_train_bin = y_train.reshape(-1, 1)
y_test_bin = y_test.reshape(-1, 1)

model_bin = LogisticRegressionScratch(lr=0.1, epochs=500, batch_size=5, multiclass=False)
model_bin.fit(X_train, y_train_bin)

y_pred_train = model_bin.predict(X_train)
y_pred_test = model_bin.predict(X_test)

print("Binary Logistic Regression:")
print("Train misclassifications:", np.sum(y_pred_train != y_train))
print("Test misclassifications:", np.sum(y_pred_test != y_test))
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

# Plot training error
plt.plot(model_bin.errors)
plt.title("Binary Logistic Regression Training Error")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.show()

y_onehot = onehot_encode(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

model_multi = LogisticRegressionScratch(lr=0.1, epochs=1000, batch_size=10, multiclass=True)
model_multi.fit(X_train, y_train)

y_pred_train = model_multi.predict(X_train)
y_pred_test = model_multi.predict(X_test)

print("\nMulticlass Logistic Regression:")
print("Train misclassifications:", np.sum(y_pred_train != np.argmax(y_train, axis=1)))
print("Test misclassifications:", np.sum(y_pred_test != np.argmax(y_test, axis=1)))
print("Train Accuracy:", accuracy_score(np.argmax(y_train, axis=1), y_pred_train))
print("Test Accuracy:", accuracy_score(np.argmax(y_test, axis=1), y_pred_test))

cm_train = confusion_matrix(np.argmax(y_train, axis=1), y_pred_train, labels=[0,1,2])
cm_test = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test, labels=[0,1,2])

print("\nConfusion Matrix - Train\n", cm_train)
print("\nConfusion Matrix - Test\n", cm_test)

plt.plot(model_multi.errors)
plt.title("Multiclass Logistic Regression Training Error")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.show()

print("\nFinal Weights (Multiclass):\n", model_multi.W)
