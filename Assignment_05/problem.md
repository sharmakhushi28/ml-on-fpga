a) Write a Naïve Bayes algorithm to classify the iris database. Use all 150 observations and all 3 classes. Use a Gaussian distribution as sown in the function below.
## mu is the mean, sig is the standard dev, x is the feature value
def PGauss(mu, sig, x):
return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.) + 1e-300) )
find the distribution (mu and stdev) for each feature for each class
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:,0:4] ## from this only take features 0,1,2,3
y = iris.target
Using this distribution find the probability of each class for each observation (product of probability of each feature given that class and the probability of the class). This may be useful code
self.u[f,c] = X[np.where(t==c),f].mean()
self.s[f,c] = X[np.where(t==c),f].std()
The class that has the highest probability is the predicted class for that observation

b) Run it for the iris database and determine
You don’t need to do standard scaling that is really built into the method, but you do need to split into training and test (or validation) as usual. 30% test should be fine.
•
Print out the number of samples (observations) in the training(or test) and the number of features used
•
the number of miss classifications,
•
the accuracy in the typical way and also
•
dump out the indices of cases that were incorrectly classified, that is the misclassifications, you can use this code
## where were there errors
err=np.where(y_train!=ypred)
print('errors at indices ', err, 'actual classificiton ', y_train[err],' pred myNB ', ypred[err])
capture all output in screen grabs
Also dump out the total number of misclassifications.

c) Train and test using the GaussianNB Naïve Bayes algorithm in Sklearn. Compare results with (2) and comment on differences.
Dump out the accuracy, number of misclassifications, in train and as well in test. Since there are multiple classes also dump out a confusion matrix for train and separately for test.
from sklearn.metrics import confusion_matrix cm = confusion_matrix(y_combined, y_pred, labels=[0, 1, 2])
print(' number in each class down vs number in each known class across ')
print(' confusion matrix \n 0 1 2\n', cm.T) ## transpose