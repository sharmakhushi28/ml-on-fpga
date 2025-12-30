# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:08:07 2025

@author: khush
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Provided Parameters and Weights ---

Image_dim = 7
stride = 2
Ydim = 3  # Output dimension of convolution
Wdim = 3  # Kernel dimension

# Convolution weights
W1 = np.array([
    [1.6975548, -0.07326141, -0.41880725],
    [0.12228276, -0.19572004, 0.81986898],
    [0.8876136, -1.8629187, -0.97661273]
])

# ANN weights
W2 = np.array([[1.10485759, 0.2120758, -1.31693339]])

# Target labels for training
target = np.array([
    [1,0,0], [1,0,0], [1,0,0],
    [0,1,0], [0,1,0], [0,1,0],
    [0,0,1], [0,0,1], [0,0,1]
], dtype=float)

# Target labels for testing
targett = np.array([[1,0,0], [0,0,1], [0,1,0]], dtype=float)


# --- Provided Helper Functions ---

# Categorizer (Quantizer)
def categorize(X):
    XX = np.zeros(len(X), float)
    # Convert to list to use .index()
    X_list = X.tolist()
    maxind = X_list.index(max(X_list))
    XX[maxind] = 1.0
    return(XX)

# 180-degree matrix rotation
def M180deg(M):
    # reverse order columns and then rows
    return(np.flip(np.flip(M, axis=1), axis=0))


# --- Training Data (Images 0-8) ---

image7by7 = np.zeros([9, Image_dim, Image_dim], float)

# Image 0
image7by7[0,1,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,2,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,4,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,5,:] = np.array([0,0,1,1,1,0,0])

# Image 1
image7by7[1,1,:] = np.array([0,0,0,1,1,0,0])
image7by7[1,2,:] = np.array([0,0,0,0,1,0,0])
image7by7[1,3,:] = np.array([0,0,0,0,1,0,0])
image7by7[1,4,:] = np.array([0,0,0,0,1,0,0])
image7by7[1,5,:] = np.array([0,0,0,1,1,1,0])

# Image 2
image7by7[2,1,:] = np.array([0,0,0,1,0,0,0])
image7by7[2,2,:] = np.array([0,0,0,0,1,0,0])
image7by7[2,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[2,4,:] = np.array([0,0,0,1,1,0,0])
image7by7[2,5,:] = np.array([0,1,1,1,1,1,0])

# Image 3
image7by7[3,1,:] = np.array([0,1,1,1,1,1,0])
image7by7[3,2,:] = np.array([0,1,0,0,0,1,0])
image7by7[3,3,:] = np.array([0,1,0,1,0,1,0])
image7by7[3,4,:] = np.array([0,1,1,0,0,1,0])
image7by7[3,5,:] = np.array([0,1,1,1,1,1,0])

# Image 4
image7by7[4,1,:] = np.array([0,0,0,1,1,0,0])
image7by7[4,2,:] = np.array([0,0,0,1,0,0,0])
image7by7[4,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[4,4,:] = np.array([0,0,1,0,0,0,0])
image7by7[4,5,:] = np.array([0,0,1,1,1,0,0])

# Image 5
image7by7[5,1,:] = np.array([0,0,1,1,1,0,0])
image7by7[5,2,:] = np.array([0,0,1,0,0,0,0])
image7by7[5,3,:] = np.array([0,0,1,1,1,0,0])
image7by7[5,4,:] = np.array([0,0,0,0,1,0,0])
image7by7[5,5,:] = np.array([0,0,1,1,1,0,0])

# Image 6
image7by7[6,1,:] = np.array([0,1,1,1,1,1,0])
image7by7[6,2,:] = np.array([0,1,0,0,0,0,0])
image7by7[6,3,:] = np.array([0,1,1,1,1,0,0])
image7by7[6,4,:] = np.array([0,0,0,0,1,1,0])
image7by7[6,5,:] = np.array([0,1,1,1,1,1,0])

# Image 7
image7by7[7,1,:] = np.array([0,0,1,1,1,0,0])
image7by7[7,2,:] = np.array([0,1,0,0,0,1,0])
image7by7[7,3,:] = np.array([0,0,0,0,1,0,0])
image7by7[7,4,:] = np.array([0,1,0,0,0,1,0])
image7by7[7,5,:] = np.array([0,0,1,1,1,0,0])

# Image 8
image7by7[8,1,:] = np.array([0,1,1,1,1,1,0])
image7by7[8,2,:] = np.array([0,1,0,0,0,1,0])
image7by7[8,3,:] = np.array([0,0,0,1,1,1,0])
image7by7[8,4,:] = np.array([0,1,0,0,0,1,0])
image7by7[8,5,:] = np.array([0,1,1,1,1,1,0])


# --- Test Data (Images 0-2) ---

image7by7t = np.zeros([3, Image_dim, Image_dim], float)

# Test Image 0
image7by7t[0,1,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,2,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,3,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,4,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,5,:] = np.array([0,0,0,1,0,0,0])

# Test Image 1
image7by7t[1,1,:] = np.array([0,0,1,1,1,0,0])
image7by7t[1,2,:] = np.array([0,1,0,0,0,1,0])
image7by7t[1,3,:] = np.array([0,0,0,0,1,1,0])
image7by7t[1,4,:] = np.array([0,1,0,0,0,1,0])
image7by7t[1,5,:] = np.array([0,0,1,1,1,0,0])

# Test Image 2
image7by7t[2,1,:] = np.array([0,0,1,1,1,1,0])
image7by7t[2,2,:] = np.array([0,0,1,0,1,0,0])
image7by7t[2,3,:] = np.array([0,0,0,1,0,0,0])
image7by7t[2,4,:] = np.array([0,0,1,0,1,0,0])
image7by7t[2,5,:] = np.array([0,0,1,1,1,1,0])


# --- Core Functions ---

def standardize(image):
    """Standardizes an image to have mean 0 and std 1."""
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        # If std is 0, image is constant. (image - mean) will be all zeros.
        return image - mean
    else:
        return (image - mean) / std

def convolve2d(image, kernel, stride):
    """Performs 2D convolution (technically correlation)."""
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    # Calculate output dimensions
    out_h = (img_h - ker_h) // stride + 1
    out_w = (img_w - ker_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for y in range(out_h):
        for x in range(out_w):
            # Get top-left corner of the current patch
            y_start = y * stride
            y_end = y_start + ker_h
            x_start = x * stride
            x_end = x_start + ker_w
            
            # Extract the patch
            patch = image[y_start:y_end, x_start:x_end]
            
            # Apply kernel: element-wise multiply and sum
            output[y, x] = np.sum(patch * kernel)
            
    return output

def ann_forward(conv_output, W2):
    """Performs the forward pass of the simple ANN."""
    # W2 is (1, 3)
    # conv_output is (3, 3)
    # ANN output = W2 @ conv_output
    # (1, 3) @ (3, 3) -> (1, 3)
    ann_output = W2 @ conv_output
    
    # Flatten to (3,) for the categorize function
    return ann_output.flatten()

def predict(image, W1_filter, W2_weights, stride):
    """Full forward propagation pipeline for a single image."""
    
    # 1. Standardize the image
    std_image = standardize(image)
    
    # 2. Rotate the kernel (as hinted in the problem)
    kernel = M180deg(W1_filter)
    
    # 3. Perform convolution
    conv_out = convolve2d(std_image, kernel, stride)
    
    # 4. Perform ANN forward pass
    scores = ann_forward(conv_out, W2_weights)
    
    # 5. Get final one-hot prediction
    prediction = categorize(scores)
    
    return prediction

# --- Run Predictions ---

print("--- 📈 Training Results ---")
train_correct = 0
for i in range(len(image7by7)):
    img = image7by7[i]
    true_label = target[i]
    pred_label = predict(img, W1, W2, stride)
    
    is_correct = np.array_equal(pred_label, true_label)
    if is_correct:
        train_correct += 1
        
    print(f"Image {i}: Pred {pred_label}, Target {true_label}, Correct: {is_correct}")

print(f"\nTraining Accuracy: {train_correct / len(image7by7):.2%}")


print("\n--- 🧪 Test Results ---")
test_correct = 0
for i in range(len(image7by7t)):
    img = image7by7t[i]
    true_label = targett[i]
    pred_label = predict(img, W1, W2, stride)
    
    is_correct = np.array_equal(pred_label, true_label)
    if is_correct:
        test_correct += 1
        
    print(f"Test Image {i}: Pred {pred_label}, Target {true_label}, Correct: {is_correct}")

print(f"\nTest Accuracy: {test_correct / len(image7by7t):.2%}")