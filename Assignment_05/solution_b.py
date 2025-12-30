# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:22:16 2025

@author: khush
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Gaussian probability function
def PGauss(mu, sig, x):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.) + 1e-300))

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :4]   
y = iris.target        

# Split into train/test sets
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.30, random_state=42, stratify=y
)

classes = np.unique(y_train)
n_classes = len(classes)
n_features = X_train.shape[1]

mu = np.zeros((n_classes, n_features))
sig = np.zeros((n_classes, n_features))
priors = np.zeros(n_classes)

for c in classes:
    X_c = X_train[y_train == c]
    mu[c, :] = X_c.mean(axis=0)
    sig[c, :] = X_c.std(axis=0)
    priors[c] = len(X_c) / float(len(X_train))

# Prediction function
def predict_naive_bayes(X_input):
    preds = []
    for x in X_input:
        probs = []
        for c in classes:
            likelihood = np.prod(PGauss(mu[c], sig[c], x))
            posterior = likelihood * priors[c]
            probs.append(posterior)
        preds.append(np.argmax(probs))
    return np.array(preds)

y_pred = predict_naive_bayes(X_test)

# Compute metrics
n_train, n_test = len(X_train), len(X_test)
n_features_used = X_train.shape[1]
mis_idx = np.where(y_test != y_pred)[0]
n_mis = len(mis_idx)
acc = accuracy_score(y_test, y_pred)

# Print results
print(f"Number of training samples: {n_train}, Number of test samples: {n_test}")
print(f"Number of features used: {n_features_used}")
print(f"Number of misclassifications: {n_mis}")
print(f"Accuracy: {acc*100:.2f}%")

print("\nMisclassified indices (within test set):", mis_idx.tolist())
print("Original dataset indices:", idx_test[mis_idx].tolist())

# Detailed misclassification info
print("\nDetails of misclassified cases:")
for i in mis_idx:
    print(f"Index {i:2d} (orig {idx_test[i]:3d}) - Actual: {y_test[i]}, Predicted: {y_pred[i]}")

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
