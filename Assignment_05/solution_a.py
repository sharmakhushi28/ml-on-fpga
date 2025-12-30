# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:11:51 2025

@author: khush
"""

import numpy as np
from sklearn import datasets

# Gaussian probability function
def PGauss(mu, sig, x):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.) + 1e-300))

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :4]  # all 4 features
y = iris.target       # class labels (0, 1, 2)
classes = np.unique(y)
n_classes = len(classes)
n_features = X.shape[1]

mu = np.zeros((n_classes, n_features))
sig = np.zeros((n_classes, n_features))
priors = np.zeros(n_classes)

for c in classes:
    X_c = X[y == c]
    mu[c, :] = X_c.mean(axis=0)
    sig[c, :] = X_c.std(axis=0)
    priors[c] = X_c.shape[0] / float(X.shape[0])  # P(class)

predictions = []
for x in X:
    probs = []
    for c in classes:
        # Product of probabilities of features given class
        likelihood = np.prod(PGauss(mu[c], sig[c], x))
        posterior = likelihood * priors[c]
        probs.append(posterior)
    # Class with highest posterior probability
    predictions.append(np.argmax(probs))

predictions = np.array(predictions)

# Accuracy
accuracy = np.mean(predictions == y)
print(f"Naïve Bayes classification accuracy: {accuracy * 100:.2f}%")
