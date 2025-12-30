# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:34:17 2025

@author: khush
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, np.arange(len(y)), test_size=0.30, random_state=42, stratify=y
)

gnb = GaussianNB()  
gnb.fit(X_tr, y_tr)

yhat_tr = gnb.predict(X_tr)
yhat_te = gnb.predict(X_te)

def report(split_name, y_true, y_pred, indices):
    acc = accuracy_score(y_true, y_pred)
    errs = np.where(y_true != y_pred)[0]
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

    print(f"\n=== {split_name} ===")
    print(f"samples: {len(y_true)} | features: {X.shape[1]}")
    print(f"misclassifications: {len(errs)}")
    print(f"accuracy: {acc*100:.2f}%")
    print("error indices within split:", errs.tolist())
    print("original dataset indices for those errors:", indices[errs].tolist())
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)
    print("\nConfusion matrix transposed (class counts down):\n", cm.T)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=iris.target_names))

report("TRAIN", y_tr, yhat_tr, idx_tr)
report("TEST",  y_te, yhat_te, idx_te)
