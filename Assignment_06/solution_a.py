# -*- coding: utf-8 -*-                                       
"""
Created on Thu Nov 13 13:57:59 2025

@author: khush
"""

import numpy as np

def _pad_for_mode(img, kH, kW, padding):
    if padding == "valid":
        return img
    elif padding == "same":
        
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        raise ValueError("padding must be 'valid' or 'same'")

def corr2d(img: np.ndarray, W: np.ndarray, stride: int = 1, padding: str = "valid") -> np.ndarray:
    """
    2D correlation (no kernel flip).
    img: 2D array (H, W)
    W:   2D kernel (kH, kW)
    stride: positive int
    padding: 'valid' or 'same'
    """
    assert img.ndim == 2 and W.ndim == 2
    kH, kW = W.shape
    x = _pad_for_mode(img, kH, kW, padding)

    H, WW = x.shape
    outH = (H - kH) // stride + 1
    outW = (WW - kW) // stride + 1
    out = np.zeros((outH, outW), dtype=float)

    for i_out, i in enumerate(range(0, H - kH + 1, stride)):
        for j_out, j in enumerate(range(0, WW - kW + 1, stride)):
            patch = x[i:i+kH, j:j+kW]
            out[i_out, j_out] = np.sum(patch * W)
    return out

def conv2d(img: np.ndarray, W: np.ndarray, stride: int = 1, padding: str = "valid") -> np.ndarray:
    """
    2D convolution = correlation with a 180°-rotated kernel.
    """
    Wrot = np.flipud(np.fliplr(W))
    return corr2d(img, Wrot, stride=stride, padding=padding)


f = np.zeros((7, 7), dtype=float)
f[3, 3] = 1.0


g = np.array([[1, 2, 3],
              [3, 4, 5],
              [6, 7, 8]], dtype=float)


corr_res = corr2d(f, g, stride=1, padding="valid")
conv_res = conv2d(f, g, stride=1, padding="valid")

print("Correlation (valid):\n", corr_res)
print("\nConvolution (valid):\n", conv_res)
