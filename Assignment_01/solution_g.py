import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

dataset = pd.read_csv("/Users/khush/Downloads/randPolyN2.csv")

df = pd.read_csv("/Users/khush/Downloads/randPolyN2.csv")
X = df.drop(columns=["Z"]).values
y = df["Z"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

degrees = [1, 2, 3, 4, 5]
train_r2 = []
test_r2 = []

for d in degrees:
        poly = PolynomialFeatures(degree=d)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
    
        model = Ridge(alpha=0.1)
        model.fit(X_train_poly, y_train)
        
        y_train_p = model.predict(X_train_poly)
        y_test_p = model.predict(X_test_poly)
                                 
        train_r2.append(r2_score(y_train, y_train_p))
        test_r2.append(r2_score(y_test, y_test_p))

        print(f"Degree {d}: Train R² = {train_r2[-1]:.3f}, Test R² = {test_r2[-1]:.3f}")

        plt.scatter(y_test, y_test_p, alpha=0.7, label=f"Degree {d}")


# Compare R² results
plt.plot(degrees, test_r2, marker="o", label="Test R²", color="red")
plt.plot(degrees, train_r2, marker="s", label="Train R²", color="blue")
plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.title("R² vs Polynomial Degree")
plt.legend()
plt.show()
