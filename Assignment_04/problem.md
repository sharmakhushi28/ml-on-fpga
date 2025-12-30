a) Write your own cross entropy cost function logistic regression algorithm. Use the iris dataset as shown below. I recommend Stochastic Gradient Descents.
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:,0:4]
y = iris.target
You can start by simplifying this to the first two sets of 50 points so there are only two classes. You can do this to develop and test your code. Then expand the code to a multiclass with all 150 points and 3 classes. To do that you will have to one-hot code target y and use argmax to determine the prediction.
Plot error versus training step iteration. Dump out the resulting weights. Dump out the number of missclassifcations in train and in test along with the accuracy. Also dump out a confusion matrix based on predictions for the test dataset.
Useful code
##
## convert to compact form, now in compact form
## the first weight is the intercept
##
Fones = np.ones(Nobs,dtype=float)
X_std_ = np.column_stack((Fones,X_std))
##
## Activation
##
def activation(Z):
act = np.ones(len(Z),float)
for i,z in enumerate(Z):
act[i] = 1./(1.+np.exp(-np.clip(z,-250,250)))
return( act )
##
## convert to multiclass one hot coding
##
def onehoty(y):
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y.reshape(len(y), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
return(onehot_encoded)
##
## Argmax function
##
def argmax(Z):
maxz = max(Z)
for j in range(len(Z)):
Z[j] = 1.0 if Z[j]>=maxz else 0
return(Z)
If you want to show the highest correlated (optional)
##
## Most Highly Correlated
##
def mosthighlycorrelated(mydataframe, numtoreport):
… in the dataanalysis code provided previously
To use this which is in the panda dataframe
import pandas as pd
irisp = pd.DataFrame(iris.data,columns=iris.feature_names)
print("Most Highly Correlated")
print(mosthighlycorrelated(irisp,Nfeatures))
print('\n',irisp.head())
Since there are multiple classes also dump out a confusion matrix for train and separately for test.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_target, y_pred, labels=[0, 1, 2])
## if your data is one hot coded you have to convert it like this
## cm = confusion_matrix(y_train_.argmax(axis=1), ypreds_train.argmax(axis=1), labels=[0, 1, 2])
print(' number in each class down vs number in each known class across ')
print(' confusion matrix \n 0 1 2\n', cm.T) ## transpose

b) Run the logistic regression algorithm in as sklearn. Compare the solution to the solution you got in problem 4. Compare weights as well as training and test accuracy.
Turn in all your code files and results organized so that the grader can easily find and test your work