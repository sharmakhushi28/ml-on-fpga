port pandas as pd
import numpy as np
dataset = pd.read_csv("/Users/khush/Downloads/regressionprob1_train0.csv")
df = pd.read_csv('/Users/khush/Downloads/regressionprob1_train0.csv')
X = df.iloc[:, 0:4].values
y = df['F'].values
ones_ = np.ones(len(y), float)
A = np.column_stack((ones_, X))

def gradient_descent(A, y, alpha=0.001, iters=10000):
    m, n = A.shape
    w = np.zeros(n)
    cost_history = []
    
    for i in range(iters):
        y_p = A @ w
        error = y_p - y
        grad = (1/m) * (A.T @ error)
        w = w - alpha * grad
        cost = (1/(2*m)) * np.sum(error**2)
        cost_history.append(cost)
    return w, cost_history
    
w_gd, cost_hist = gradient_descent(A, y, alpha=0.001, iters=50000)    
intercept_gd = w_gd[0]
coefs_gd = w_gd[1:] 
#print("Intercept: ", intercept_gd)
#print("Coefficients: ", coefs_gd)
y_p_gd = A @ w_gd
R2_gd = r2_score(y, y_p_gd)
print("R2: ", R2_gd)
#compare matrix
ATA = A.T @ A
ATy = A.T @ y
w_exact = np.linalg.inv(ATA) @ ATy
print("matrix weights: ", w_exact)
print("gradient weights: ", w_gd)
diff = np.linalg.norm(w_exact - w_gd)
print("Difference: ", diff)
