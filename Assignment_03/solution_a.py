# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:42:43 2025

@author: khush
"""

import numpy as np

a = 20
b = np.array([-1, 1]).reshape(2, 1)
C = np.array([[2, 0], [0, 1]]).reshape(2, 2)

def f(w):
    return a + np.dot(b.T, w) + np.dot(w.T, np.dot(C, w))


def grad_des(g, w, alpha, iter, tol):
    tbeg = np.datetime64('now')
    count = 1
    Dg = grad(w)  
    updatew = 1.0
    errors = []

    while (count < iter) and (np.linalg.norm(updatew) > tol):
        gradient = g(w)
        Dg = np.linalg.norm(gradient) 
        updatew = -alpha * gradient
        w = w + updatew
        errors.append(np.linalg.norm(updatew))
        count += 1
    
    tend = np.datetime64('now')
    dtime = (tend - tbeg).astype('timedelta64[ms]').item().total_seconds() * 1000

    return w, count, updatew, dtime

def grad(w):
    return b + 2 * np.dot(C, w)

if __name__ == '__main__':
   
    w0 = np.array([[-0.3], [0.2]])
    learning_rate = 0.1
    max_iterations = 1000
    tolerance = 1e-6

    f_at_w0 = f(w0)
    print(f"Value of f(w) at w0 = {w0.T[0]}:")
    print(f"f(w0) = {f_at_w0[0][0]:.4f}\n")

    grad_at_w0 = grad(w0)
    print("Value of the gradient at w0:")
    print(f"grad(w0) = {grad_at_w0.T[0]}\n")

    print("--- Running Gradient Descent ---")
    
    w = w0.copy() 
    
    w_final, count, last_update, dtime = grad_des(grad, w, learning_rate, max_iterations, tolerance)
    
    print('Gradient descents constant alpha, iter ', count)
    print(f'delay {dtime:.3f} ms, time/iteration {dtime/count:.3f} ms')
    print('learning rate alpha %.3g ' % learning_rate)
    print(f"Final weights (w): {w_final.T[0]}")
    print(f"Last update: {np.linalg.norm(last_update):.6f}")