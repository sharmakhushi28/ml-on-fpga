# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:46:42 2025

@author: khush
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

X = np.array([
    [-2, 4, -1],
    [ 4, 1, -1],
    [ 1, 6, -1],
    [ 2, 4, -1],
    [ 5, 6, -1]
])
y = np.array([-1, -1, 1, 1, 1])

def relu(x):
    return np.maximum(0, x)
def quantizer(x):
    return np.where(x > 0, 1, -1)
def perceptron_relu(X, y, eta=0.1, epochs=20):
    w = np.ones(X.shape[1])   
    errors = []

    for epoch in range(epochs):
        total_error = 0
        print(f"\nEpoch {epoch+1}:")
        for i in range(len(X)):
            net = np.dot(X[i], w)
            out = relu(net)               
            y_pred = quantizer(out)        
            error = y[i] - y_pred          
            update = eta * error * X[i]    
            w = w + update

            total_error += abs(error)

            print(f"  Sample {i}: net={net:.2f}, pred={y_pred}, "
                  f"actual={y[i]}, error={error}, weights={w}")

        errors.append(total_error)

        if total_error == 0:
            break
    return w, errors
eta = 0.1
epochs = 20
w, errors = perceptron_relu(X, y, eta, epochs)

print("\nFinal weights:", w)
Ypred = []
for i in range(len(X)):
    net = np.dot(X[i], w)
    out = relu(net)
    ypred = quantizer(out)
    Ypred.append(ypred)
    print(f"Final Prediction: x={X[i]}, net={net:.2f}, pred={ypred}, actual={y[i]}")

Ypred = np.array(Ypred)
accuracy = accuracy_score(y, Ypred) * 100
misclassified = np.sum(Ypred != y)

print("\nAccuracy: %.2f%%" % accuracy)
print("Number of misclassified cases:", misclassified)
plt.plot(errors, marker='o')
plt.title("Error vs Iteration")
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.grid()
plt.show()