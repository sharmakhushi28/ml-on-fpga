# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 22:50:12 2025

@author: khush
"""

import numpy as np
import time

a = 20
b = np.array([-1, 1]).reshape(2, 1)
C = np.array([[2, 0], [0, 1]]).reshape(2, 2)
lambda_val = 0.5 

def f(w):
    """
    Evaluates the function f(w) = a + b^T w + w^T C w + lambda * |w|^2.
    w must be a 2x1 numpy array.
    """
    regularization_term = lambda_val * np.sum(w**2)
    return a + np.dot(b.T, w) + np.dot(w.T, np.dot(C, w)) + regularization_term

def grad_analytical(w):
    """
    Evaluates the analytical derivative (gradient) of the regularized f(w).
    The gradient is b + 2Cw + 2*lambda*w.
    w must be a 2x1 numpy array.
    """
    return b + 2 * np.dot(C, w) + 2 * lambda_val * w

def grad_des_momentum(g, w, alpha, beta, iter, tol):
    """
    Performs steepest descent with momentum.
    """
    tbeg = time.process_time()
    count = 1
    h = np.zeros_like(w)  
    updatew = np.array([[1.0], [1.0]])
    errors = []

    while (count < iter) and (np.linalg.norm(updatew) > tol):
        gradient = g(w)
        h = beta * h + (1 - beta) * gradient  # Update momentum term
        updatew = -alpha * h
        w = w + updatew
        errors.append(np.linalg.norm(updatew))
        count += 1
    
    tend = time.process_time()
    dtime = (tend - tbeg) * 1000

    return w, count, np.linalg.norm(updatew), dtime, errors

def grad_des_constant_alpha(g, w, alpha, iter, tol):
    """
    Performs gradient descent with a constant learning rate.
    
    Args:
        g: The gradient function.
        w: The initial weight vector.
        alpha: The constant learning rate.
        iter: The maximum number of iterations.
        tol: The convergence tolerance.
    """
    tbeg = time.process_time()
    count = 1
    updatew = np.array([[1.0], [1.0]])
    errors = []

    while (count < iter) and (np.linalg.norm(updatew) > tol):
        gradient = g(w)
        updatew = -alpha * gradient
        w = w + updatew
        errors.append(np.linalg.norm(updatew))
        count += 1
    
    tend = time.process_time()
    dtime = (tend - tbeg) * 1000

    return w, count, np.linalg.norm(updatew), dtime, errors

def find_optimal_alpha(method_func, g, initial_w, alphas_to_test, max_iter, tol, **kwargs):
    """Helper function to find the optimal alpha for a given method."""
    optimal_alpha = None
    print("Searching for optimal initial learning rate...")
    for alpha in alphas_to_test:
        w_final, count, last_update_norm, dtime, errors = method_func(
            g, initial_w.copy(), alpha, max_iter, tol, **kwargs)
        
        if last_update_norm <= tol:
            print(f"Alpha_initial = {alpha}: Converged in {count} iterations.")
            optimal_alpha = alpha
            break
        else:
            print(f"Alpha_initial = {alpha}: Did not converge.")
    return optimal_alpha

if __name__ == '__main__':
 
    w0 = np.array([[-0.3], [0.2]])
    max_iter = 50000
    tol = 1e-6
    
    ModelResults = []

    print("\n--- Running Steepest Descent with Momentum ---")
    optimal_alpha_momentum = 10
    beta = 0.75

    w_final, count, last_update_norm, dtime, errors = grad_des_momentum(
        grad_analytical, w0.copy(), optimal_alpha_momentum, beta, max_iter, tol)
    
    momentum_results = {
        "method": "Steepest Descent with Momentum",
        "optimal_alpha": optimal_alpha_momentum,
        "beta": beta,
        "final_weights": w_final.T[0].tolist(),
        "total_iterations": count,
        "total_time_ms": dtime,
        "time_per_iter_ms": (dtime / count)
    }
    ModelResults.append(momentum_results)

    print("\n--- Running Gradient Descent with Constant Alpha ---")
    alphas_to_test_constant = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    optimal_alpha_constant = find_optimal_alpha(
        grad_des_constant_alpha, grad_analytical, w0, alphas_to_test_constant, max_iter, tol)

    if optimal_alpha_constant is not None:
        w_final, count, last_update_norm, dtime, errors = grad_des_constant_alpha(
            grad_analytical, w0.copy(), optimal_alpha_constant, max_iter, tol)
        
        constant_results = {
            "method": "Constant Alpha",
            "optimal_alpha": optimal_alpha_constant,
            "beta": None,
            "final_weights": w_final.T[0].tolist(),
            "total_iterations": count,
            "total_time_ms": dtime,
            "time_per_iter_ms": (dtime / count)
        }
        ModelResults.append(constant_results)

    print("\n--- Compiled Model Results ---")
    for result in ModelResults:
        print(f"Method: {result['method']}")
        print(f"  - Optimal Alpha: {result['optimal_alpha']}")
        if result['beta'] is not None:
            print(f"  - Beta: {result['beta']}")
        print(f"  - Final Weights: {result['final_weights']}")
        print(f"  - Total Iterations: {result['total_iterations']}")
        print(f"  - Total Time (ms): {result['total_time_ms']:.3f}")
        print(f"  - Time per Iteration (ms): {result['time_per_iter_ms']:.6f}")
        print("-" * 20)
