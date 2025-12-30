# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 20:15:16 2025

@author: khush
"""

import numpy as np
import time
import matplotlib.pyplot as plt

a = 20
b = np.array([-1, 1]).reshape(2, 1)
C = np.array([[2, 0], [0, 1]]).reshape(2, 2)

def f(w):
    """
    Evaluates the function f(w) = a + b^T w + w^T C w.
    w must be a 2x1 numpy array.
    """
    return a + np.dot(b.T, w) + np.dot(w.T, np.dot(C, w))

def grad_analytical(w):
    """
    Evaluates the analytical derivative (gradient) of f(w).
    The gradient is b + 2Cw.
    w must be a 2x1 numpy array.
    """
    return b + 2 * np.dot(C, w)

def grad_des_decreasing_alpha(g, w, alpha_initial, iter, tol):
    """
    Performs gradient descent with a learning rate that decreases with each iteration.
    alpha_initial / count.
    
    Args:
        g: The gradient function.
        w: The initial weight vector.
        alpha_initial: The initial learning rate.
        iter: The maximum number of iterations.
        tol: The tolerance for convergence.
    
    Returns:
        w: The final weights.
        count: The number of iterations to converge.
        last_update_norm: The norm of the last weight update.
        dtime: The total time for convergence in milliseconds.
        errors: A list of the norms of the weight updates at each iteration.
    """
    tbeg = time.process_time()
    count = 1
    updatew = np.array([[1.0], [1.0]])  # Initialize with a non-zero value
    errors = []
    
    while (count < iter) and (np.linalg.norm(updatew) > tol):
        alpha = alpha_initial / count  # Decreasing learning rate
        gradient = g(w)
        updatew = -alpha * gradient
        w = w + updatew
        errors.append(np.linalg.norm(updatew))
        count += 1
    
    tend = time.process_time()
    dtime = (tend - tbeg) * 1000
    
    return w, count, np.linalg.norm(updatew), dtime, errors

def run_decreasing_alpha_method():
    print("--- Running Gradient Descent with Decreasing Alpha ---")
    
    initial_w = np.array([[-0.3], [0.2]])
    tol = 1e-6
    max_iter = 50000

    alphas_to_test = [100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
    
    print("Searching for optimal initial learning rate...")
    optimal_alpha = None
    
    for alpha in alphas_to_test:
        w_final, count, last_update_norm, dtime, errors = grad_des_decreasing_alpha(
            grad_analytical, initial_w.copy(), alpha, max_iter, tol)
        
        if last_update_norm <= tol:
            print(f"Alpha_initial = {alpha}: Converged in {count} iterations.")
            optimal_alpha = alpha
            break
        else:
            print(f"Alpha_initial = {alpha}: Did not converge.")

    if optimal_alpha is not None:
        w_final, count, last_update_norm, dtime, errors = grad_des_decreasing_alpha(
            grad_analytical, initial_w.copy(), optimal_alpha, max_iter, tol)
        
        print("\n--- Results for Optimal Learning Rate ---")
        print(f"Optimal Initial Learning Rate: {optimal_alpha}")
        print(f"Final Weights: {w_final.T[0]}")
        print(f"Final Weight Update (norm): {last_update_norm:.6f}")
        print(f"Total Iterations to Converge: {count}")
        print(f"Total Time for Convergence: {dtime:.3f} ms")
        print(f"Time Delay per Iteration: {(dtime / count):.6f} ms")
        
        plt.figure()
        plt.plot(np.log(errors))
        plt.title('Log(Error) vs Iteration (Decreasing Alpha)')
        plt.xlabel('Iteration')
        plt.ylabel('Log(Error Norm)')
        plt.grid(True)
        plt.show()
    else:
        print("\nCould not find a converging learning rate in the tested range.")

if __name__ == '__main__':
    run_decreasing_alpha_method()