# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:51:23 2025

@author: khush
"""

import numpy as np

# =========================================================
# Utilities from the handout (names kept consistent)
# =========================================================

def categorize(X):
    """Quantizer: set the max entry to 1 and others to 0."""
    XX = np.zeros(len(X), float)
    maxi = X.tolist().index(max(X.tolist()))
    XX[maxi] = 1.0
    return XX

def standardize(img):
    """Per-image standardization (mean 0, std 1)."""
    mu, sd = img.mean(), img.std()
    return (img - mu) / (sd + 1e-12)

def conv2D(image, W, stride=1, Conv=True):
    """
    Square 2D corr/conv with arbitrary stride, VALID padding.
    Conv=True => true convolution (kernel rotated 180°).
    Conv=False => correlation (no flip).
    """
    kH, kW = W.shape
    if Conv:
        K = np.flipud(np.fliplr(W))  # convolution = corr with 180° kernel flip
    else:
        K = W.copy()

    H, Ww = image.shape
    outH = (H - kH) // stride + 1
    outW = (Ww - kW) // stride + 1
    y = np.zeros((outH, outW), float)

    oh = 0
    for i in range(0, H - kH + 1, stride):
        ow = 0
        for j in range(0, Ww - kW + 1, stride):
            patch = image[i:i+kH, j:j+kW]
            y[oh, ow] = np.sum(patch * K)
            ow += 1
        oh += 1
    return y

# =========================================================
# Forward pass for the CNN (7x7 -> 3x3 via stride=2) + ANN
# =========================================================

def forward_one(img7x7, W1, W2, stride=2, use_convolution=True):
    """
    img7x7: (7,7)
    W1: (3,3) conv/corr kernel
    W2: (3,) ANN weights applied as X2 @ W2
    Returns: (X1, X2, X3) = (standardized, feature map, logits)
    """
    X1 = standardize(img7x7)
    X2 = conv2D(X1, W1, stride=stride, Conv=use_convolution)  # (3,3)
    X3 = X2 @ W2                                              # (3,)
    return X1, X2, X3

def predict_batch(images, W1, W2, stride=2, use_convolution=True):
    preds = []
    for im in images:
        _, _, X3 = forward_one(im, W1, W2, stride=stride, use_convolution=use_convolution)
        preds.append(categorize(X3))
    return np.vstack(preds)

def accuracy(pred_onehots, tgt_onehots):
    return (pred_onehots.argmax(1) == tgt_onehots.argmax(1)).mean()

# =========================================================
# Backprop (Part 3) from scratch — squared error loss
# =========================================================

def grads_one(img, tgt, W1, W2, stride=2, use_convolution=True):
    """
    L = ||X3 - tgt||^2
    Returns dL/dW1 (3x3), dL/dW2 (3,)
    """
    X1, X2, X3 = forward_one(img, W1, W2, stride=stride, use_convolution=use_convolution)

    # dL/dX3 (3,)
    dX3 = 2.0 * (X3 - tgt)

    # dL/dW2 = X2^T @ dX3 -> (3,)
    gW2 = X2.T @ dX3

    # dL/dX2 = outer(dX3, W2) -> (3,3)
    dX2 = np.outer(dX3, W2)

    # dL/dW1: correlate upstream (3x3) grads with the corresponding 3x3 patches of X1
    gW1 = np.zeros_like(W1)
    for i in range(3):
        for j in range(3):
            patch = X1[i*stride:i*stride+3, j*stride:j*stride+3]
            gW1 += dX2[i, j] * patch
    return gW1, gW2

def train_cnn_ann(images, targets, lr=0.05, epochs=600, stride=2, use_convolution=True, seed=1, verbose_every=50):
    """
    Initializes W1, W2 like the handout (random ~U(-0.5,0.5)),
    then does simple batch gradient descent.
    """
    np.random.seed(seed)
    W1 = (np.random.rand(3,3) - 0.5)
    W2 = (np.random.rand(3) - 0.5)

    N = len(images)
    for ep in range(1, epochs+1):
        g1 = np.zeros((3,3)); g2 = np.zeros(3); loss = 0.0
        for im, tgt in zip(images, targets):
            # accumulate loss
            _, _, X3 = forward_one(im, W1, W2, stride=stride, use_convolution=use_convolution)
            loss += np.sum((X3 - tgt)**2)

            # grads
            dW1, dW2 = grads_one(im, tgt, W1, W2, stride=stride, use_convolution=use_convolution)
            g1 += dW1; g2 += dW2

        g1 /= N; g2 /= N; loss /= N
        W1 -= lr * g1
        W2 -= lr * g2

        if verbose_every and ep % verbose_every == 0:
            print(f"epoch {ep:4d}  loss {loss:.6f}")

    return W1, W2

# =========================================================
# ---------- PASTE YOUR DATA FROM THE HANDOUT HERE ----------
# (Keep shapes consistent; these are placeholders.)
# =========================================================

# Image and label dimensions (per handout)
Image_dim = 7
stride = 2
Ydim = 3      # 3 classes
Wdim = 3      # conv kernel is 3x3

# --- TRAINING TARGETS (one-hots) ---
# Replace with the exact 'target = np.array([...],float)' from the PDF:
# target = np.array([...], float)

# --- TRAINING IMAGES ---
# Replace with the exact image7by7[...] assignments from the PDF to build a
# numpy array 'image7by7' of shape (N, 7, 7).
# Example placeholder:
# image7by7 = np.zeros((9,7,7), float)
# image7by7[...] = ...  # paste your blocks

# --- TEST TARGETS ---
# Replace with 'targett = np.array([...], float)' from the PDF:
# targett = np.array([...], float)

# --- TEST IMAGES ---
# Replace with the exact image7by7t[...] blocks from the PDF to build
# 'image7by7t' of shape (M, 7, 7).
# image7by7t = np.zeros((3,7,7), float)
# image7by7t[...] = ...

# =========================================================
# Part 2: forward with provided weights (from the prompt)
# =========================================================
W1_given = np.array([[ 1.6975548 , -0.07326411, -0.41880725],
                     [ 0.12282276, -0.19572004,  0.81896898],
                     [ 0.8876516 , -1.8629187 , -0.97561273]], float)
W2_given = np.array([ 1.10485759,  0.2102758 , -1.3169339], float)

def run_part2():
    preds_train = predict_batch(image7by7, W1_given, W2_given, stride=stride, use_convolution=True)
    acc_train = accuracy(preds_train, target)
    print("Part 2 — Train accuracy with provided weights:", acc_train)

    preds_test = predict_batch(image7by7t, W1_given, W2_given, stride=stride, use_convolution=True)
    acc_test = accuracy(preds_test, targett)
    print("Part 2 — Test  accuracy with provided weights:", acc_test)

# =========================================================
# Part 3: learn the weights with backprop (from scratch)
# =========================================================
def run_part3(lr=0.05, epochs=600, use_convolution=True):
    W1_learned, W2_learned = train_cnn_ann(
        image7by7, target, lr=lr, epochs=epochs, stride=stride,
        use_convolution=use_convolution, seed=1, verbose_every=50
    )
    print("\nLearned W1:\n", W1_learned)
    print("\nLearned W2:\n", W2_learned)

    preds_train = predict_batch(image7by7, W1_learned, W2_learned, stride=stride, use_convolution=use_convolution)
    print("Part 3 — Train accuracy:", accuracy(preds_train, target))

    preds_test = predict_batch(image7by7t, W1_learned, W2_learned, stride=stride, use_convolution=use_convolution)
    print("Part 3 — Test  accuracy:", accuracy(preds_test, targett))

# =========================================================
# How to run after you pasted the arrays:
# =========================================================
# run_part2()         # evaluate with the given W1/W2
# run_part3()         # learn W1/W2 from scratch
