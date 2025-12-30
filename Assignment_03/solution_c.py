# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:01:01 2025

@author: khush
"""

import numpy as np
import matplotlib.pyplot as plt
import time

a = 20
b = np.array([[-1.0], [1.0]])
C = np.array([[2.0, 0.0],
              [1.0, 1.0]])

def f(w):
    return a + np.dot(b.T, w) + np.dot(w.T, np.dot(C, w))

def grad_f(w):
    return b + (C + C.T).dot(w)

def grad_descent_decreasing_lr(f, grad_f, w0, alpha=0.1, max_iter=1000, tol=1e-6):
    t_start = time.process_time()
    w = w0.copy()
    errors = []
    count = 1
    while count <= max_iter:
        g = grad_f(w)
        step_size = alpha / count
        update = -step_size * g
        w = w + update
        err = np.linalg.norm(update)
        errors.append(err)
        if err < tol:
            break
        count += 1
    t_end = time.process_time()
    return {
        "method": "Decreasing LR (alpha/count)",
        "alpha": alpha,
        "final_w": w,
        "iterations": count,
        "final_update": err,
        "total_time_ms": (t_end - t_start) * 1000,
        "errors": errors,
        "converged": err < tol
    }

def grad_descent_normalized(f, grad_f, w0, alpha=0.01, max_iter=1000, tol=1e-6):
    t_start = time.process_time()
    w = w0.copy()
    errors = []
    count = 1
    while count <= max_iter:
        g = grad_f(w)
        g_norm = np.linalg.norm(g)
        if g_norm == 0:   # avoid division by zero
            break
        update = -alpha * g / g_norm   # normalized step
        w = w + update
        err = np.linalg.norm(update)
        errors.append(err)
        if err < tol:
            break
        count += 1
    t_end = time.process_time()
    return {
        "method": "Normalized Gradient Descent",
        "alpha": alpha,
        "final_w": w,
        "iterations": count,
        "final_update": err,
        "total_time_ms": (t_end - t_start) * 1000,
        "errors": errors,
        "converged": err < tol
    }

w0 = np.array([[-0.3], [0.2]])

results1 = grad_descent_decreasing_lr(f, grad_f, w0, alpha=1.0, max_iter=500)
results2 = grad_descent_normalized(f, grad_f, w0, alpha=0.05, max_iter=500)

ModelResults = [results1, results2]

for res in ModelResults:
    print("\nMethod:", res["method"])
    print("  Alpha:", res["alpha"])
    print("  Final w:", res["final_w"].flatten())
    print("  Iterations:", res["iterations"])
    print("  Final update:", res["final_update"])
    print("  Total time (ms):", res["total_time_ms"])
    print("  Converged:", res["converged"])

plt.figure(figsize=(8,5))
plt.plot(np.log(results1["errors"]), label=results1["method"])
plt.plot(np.log(results2["errors"]), label=results2["method"])
plt.xlabel("Iteration")
plt.ylabel("log(Error)")
plt.title("Comparison of Gradient Methods")
plt.legend()
plt.grid(True)
plt.show()
