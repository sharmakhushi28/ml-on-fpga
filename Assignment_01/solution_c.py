from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

dataset = pd.read_csv("/Users/khush/Downloads/regressionprob1_test0.csv")

df = pd.read_csv("/Users/khush/Downloads/regressionprob1_test0.csv")

X_train = df.iloc[:,0:4].values
Y_train = df['F'].values
A = np.column_stack((np.ones(len(X_train)), X_train))

A_train = np.column_stack((np.ones(len(Y_train)), X_train))

ATA = A.T @A
ATY = A.T @Y_train


test_data = pd.read_csv("/Users/khush/Downloads/regressionprob1_test0.csv")
X_test = test_data.iloc[:,0:4].values
Y_test = test_data['F'].values

A_test = np.column_stack((np.ones(len(Y_test)), X_test))
w = np.linalg.solve(ATA, ATY)
Yp = A @ w

R2_test = r2_score(Y_test, Yp)

print("weights: ", w)
print("R2 Train: ", r2_score)
print("R2 Test: ",R2_test)
