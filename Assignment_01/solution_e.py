from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
dataset = pd.read_csv("/Users/khush/Downloads/regressionprob1_train0.csv")
df = pd.read_csv("/Users/khush/Downloads/regressionprob1_train0.csv")
X = df.iloc[:, 0:4].values
y = df['F'].values
ones_ = np.ones(len(y), float)
A = np.column_stack((ones_, X))
ATA = A.T @ A
ATy = A.T @ y
w_exact = np.linalg.inv(ATA) @ ATy
y_p_exact = A @ w_exact
R2_exact = r2_score(y, y_p_exact)
print("matrix inversion")
print("intercept: ", w_exact[0])
print("Coefficients: ", w_exact[1:])
print("R²: ", R2_exact)
def gradient_descent(A, y, alpha=0.001, iters=50000):
    m, n = A.shape
    w = np.zeros(n)
    for i in range(iters):
        y_p = A @ w
        error = y_p - y
        grad = (1/m) * (A.T @ error)
        w -= alpha * grad
    return w
w_gd = gradient_descent(A, y)
y_p_gd = A @ w_gd
R2_gd = r2_score(y, y_p_gd)
print("gradient descent")
print("intercept: ", w_gd[0])
print("Coefficients: ", w_gd[1:])
print("R²: ", R2_gd)
model = LinearRegression(fit_intercept = True)
model.fit(X, y)
intercept_sklearn = model.intercept_
coefs_sklearn = model.coef_
y_p_sklearn = model.predict(X)
R2_sklearn = r2_score(y, y_p_sklearn)
print("Sklearn Intercept:", intercept_sklearn)
print("Sklearn Coefficients:", coefs_sklearn)
print("Training R² (Sklearn):", R2_sklearn)
# part 2
print("comparison of Intercept")
print("matrix inversion: ", w_exact[0])
print("gradient descent: ", w_gd[0])
print("Sklearn: ", intercept_sklearn)
print("comparison of Coefficient")
print("matrix inversion: ", w_exact[1:])
print("gradient descent: ", w_gd[1:])
print("Sklearn: ", coefs_sklearn)
print("comparison of Training R² ")
print("matrix inversion: ", r2_score(y, A @ w_exact))
print("gradient descent: ", R2_gd)
print("Sklearn: ", R2_sklearn)
