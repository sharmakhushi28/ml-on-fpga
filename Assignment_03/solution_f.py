# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:25:27 2025

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

H = (C + C.T)
L = max(np.linalg.eigvals(H)).real 

z = (1 + np.sqrt(5)) / 2 

def goldendelta(x4, x1, z):
    return (x4 - x1) / z

def goldensearch(g, w, h, x1, x4, accuracy=1e-6):
    x2 = x4 - goldendelta(x4, x1, z)
    x3 = x1 + goldendelta(x4, x1, z)
    f1 = g(w - x1 * h); f2 = g(w - x2 * h)
    f3 = g(w - x3 * h); f4 = g(w - x4 * h)
    i = 0
    error = abs(x4 - x1)
    while error > accuracy:
        if (f2 < f3):
            x4, f4 = x3, f3
            x3, f3 = x2, f2
            x2 = x4 - goldendelta(x4, x1, z)
            f2 = g(w - x2 * h)
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3 = x1 + goldendelta(x4, x1, z)
            f3 = g(w - x3 * h)
        i += 1
        error = abs(f4 - f1)
    return (x1 + x4) / 2.0, i, error

def golden(g, w, h, alpha):
    alpha, iters, error = goldensearch(g, w, h, alpha/10., alpha*10.0, 1e-6)
    return alpha

def grad_descent_momentum(f, grad_f, w0, alpha=1.0, beta=0.5, max_iter=1000, tol=1e-6):
    t_start = time.process_time()
    w = w0.copy()
    errors = []
    count = 1
    h = np.zeros_like(w)  

    while count <= max_iter:
        grad = grad_f(w)
      
        h = beta * h + (1 - beta) * grad
        g = lambda x: f(x) 
       
        alpha_opt = golden(g, w, h, alpha)
        update = -alpha_opt * h
        w = w + update

        err = np.linalg.norm(update)
        errors.append(err)
        if err < tol:
            break
        count += 1

    t_end = time.process_time()
    return {
        "method": f"Momentum Steepest Descent (beta={beta})",
        "alpha": alpha,
        "beta": beta,
        "final_w": w,
        "iterations": count,
        "final_update": err,
        "total_time_ms": (t_end - t_start) * 1000,
        "errors": errors,
        "converged": err < tol
    }

w0 = np.array([[-0.3], [0.2]])

results_m0 = grad_descent_momentum(f, grad_f, w0, alpha=1.0, beta=0.0)
results_m05 = grad_descent_momentum(f, grad_f, w0, alpha=1.0, beta=0.5)
results_m075 = grad_descent_momentum(f, grad_f, w0, alpha=1.0, beta=0.75)

ModelResults = [results_m0, results_m05, results_m075]

for res in ModelResults:
    print("\nMethod:", res["method"])
    print("  Alpha:", res["alpha"])
    print("  Beta:", res["beta"])
    print("  Final w:", res["final_w"].flatten())
    print("  Iterations:", res["iterations"])
    print("  Final update:", res["final_update"])
    print("  Total time (ms):", res["total_time_ms"])
    print("  Converged:", res["converged"])

plt.figure(figsize=(8,5))
for res in ModelResults:
    plt.plot(np.log(res["errors"]), label=res["method"])
plt.xlabel("Iteration")
plt.ylabel("log(Error)")
plt.title("Momentum + Steepest Descent Comparison")
plt.legend()
plt.grid(True)
plt.show()
