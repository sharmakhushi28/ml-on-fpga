import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
dataset = pd.read_csv("/Users/khush/Downloads/randPolyN2.csv")
df = pd.read_csv("/Users/khush/Downloads/randPolyN2.csv")
X = df.drop(columns=["Z"]).values
y = df["Z"].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_std, y_train)
y_p_lr = lr.predict(X_test_std)
R2_lr = r2_score(y_test, y_p_lr)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_std, y_train)
y_p_lasso = lasso.predict(X_test_std)
R2_lasso = r2_score(y_test, y_p_lasso)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_std, y_train)
y_p_ridge = ridge.predict(X_test_std)
R2_ridge = r2_score(y_test, y_p_ridge)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  
elastic.fit(X_train_std, y_train)
y_p_elastic = elastic.predict(X_test_std)
R2_elastic = r2_score(y_test, y_p_elastic)
print("R² Scores on Test Set: ")
print("Linear Regression: ", R2_lr)
print("Lasso: ", R2_lasso)
print("Ridge: ", R2_ridge)
print("Elastic Net: ", R2_elastic)
plt.figure(figsize=(12, 10))
# Linear Regression
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_p_lr, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f"Linear Regression (R²={R2_lr:.3f})")
plt.xlabel("Actual Z")
plt.ylabel("Predicted Z")
# Lasso
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_p_lasso, alpha=0.7, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f"Lasso Regression (R²={R2_lasso:.3f})")
plt.xlabel("Actual Z")
plt.ylabel("Predicted Z")
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_p_lasso, alpha=0.7, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f"Lasso Regression (R²={R2_lasso:.3f})")
plt.xlabel("Actual Z")
plt.ylabel("Predicted Z")

# Ridge
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_p_ridge, alpha=0.7, color="orange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f"Ridge Regression (R²={R2_ridge:.3f})")
plt.xlabel("Actual Z")
plt.ylabel("Predicted Z")

# Elastic Net
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_p_elastic, alpha=0.7, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f"Elastic Net (R²={R2_elastic:.3f})")
plt.xlabel("Actual Z")
plt.ylabel("Predicted Z")

plt.tight_layout()
plt.show()


