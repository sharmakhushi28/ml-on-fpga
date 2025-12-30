Use the dataset regressionprob1_train0.csv. The column labeled F is the
target.
2X2 matrix inversion

a) Using a matrix inversion (such as shown above or Machine Learning Refined (MLR) p. 108 eqn 5.17) find the weights and intercept to predict F from the other columns. Determine the residual squared, or R2. Use compact notation as described in the lecture so that you get an intercept b (or w0). You can read the file in using pandas and create A in compact notation using this code:
import numpy as np
import pandas as pd
df = pd.read_csv('regressionprob1_train0.csv')
X = df.iloc[:,0:4].values
y = df['F'].values
ones_ = np.ones(len(y),float)
## compact notation
A = np.column_stack((ones_,X))
This Python code will calculate R2
##
## unexplained variance squared
## or R squared
## Y is the target value
## Yp is the predicted value
##
import numpy as np
def Rsquared(Y,Yp):
V = Y-Yp
Ymean = np.average(Y)
totvar = np.sum((Y-Ymean)**2)
unexpvar = np.sum(np.abs(V**2))
R2 = 1-unexpvar/totvar
return(R2)
you can alternately use
from sklearn.metrics import r2_score
R2 = r2_score(y,Yp)

b) Solve again using a linear solver like numpy.linalg.solve and compare results.

c) Now use the model trained in a) to do prediction on dataset regressionprob1_test0.csv. The column labeled F is the target. Compare these predictions to actual target labels by determining the residual squared, or R2. What can you conclude about your trained model from a) and from b)?

d) Now write your own Gradient descents code and code the gradient to find the weights to minimize the cost function in (3) using training data regressionprob1_train0.csv. (hint: see W1 Lecture 5 slide 6, 23-30 and W2 Lecture 4 slide 25) Solve for the weights and intercept using this using this training dataset. Compare this solution with your solution from (4). Be sure to use compact notation.

e) Use a canned linear regression similar to sklearn.linear_model.LinearRegression and extract the weights for the training dataset. Apply this model to the training dataset. Calculate and compare your resulting R2 and model weights to those in (4), and (5). You can use compact notation, or you can tell your regression to extract an intercept.

f) Now use the linear regression, Lasso, Ridge, and Elastic net
Use the dataset randPolyN2.csv The column labeled Z is the target.
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
model = clf.fit(X_train_std, Y_train)
from sklearn.linear_model import Ridge
clf = Ridge(alpha=0.1)
model = clf.fit(X_train_std, Y_train)
from sklearn.linear_model import ElasticNet
clf = ElasticNet(alpha=0.1)
model = clf.fit(X_train_std, Y_train)
Show your code and R2 results. Make scatterplots of each and compare.

g)  Now try each with polynomial fitting with degrees 1-5
Use the dataset randPolyN2.csv
Using this example for a Ridge Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
def create_model(degree):
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
Show your code and R2 results. Compare and draw conclusions about what this shows.
