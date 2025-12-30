a) Classify using Logistic Regression using the Sklearn library.
This is the way I analyzed the results, and you should do something similar for 2),3) and 4)
print('Number in train ',len(y_train))
y_pred = lr.predict(X_train)
mc_train = (y_train != y_pred).sum()
print('Misclassified samples: %d' % mc_train)
acc_train = accuracy_score(y_train, y_pred)
print('Accuracy: %.2f' % acc_train)
#
print('Number in test ',len(y_test))
y_pred = lr.predict(X_test)
mc_test = (y_test != y_pred).sum()
print('Misclassified samples: %d' % mc_test)
acc_test = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % acc_test)

b) Classify using SVM linear kernel, done to be comparable as well.

c) LGBM using CPU
LGBM GPU we won’t do this one because RC doesn’t have it working correctly
This is the way I did it ---
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
tbeg = time.time()
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)
## defaults num_leaves = 31,
params = {'force_col_wise': True, 'boosting_type': 'gbdt', 'num_iterations': 100, 'n_estimators': 100,
'max_depth': 5, 'num_leaves': 100, 'feature_fraction': 0.75,
'bagging_fraction': 0.75, 'bagging_freq': 1, 'lambda': 0.5, 'random_state': 3}
model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], verbose_eval=20)
print('Number in train ',len(y_train))
print('Number in train ',len(y_train))
y_train_pred = model.predict(X_train)
y_pred = np.where(y_train_pred<0.5,0,1)
tend = time.time()

d) Three layer, which would be input, hidden, and output (that is 2 NN) Neural network using Pytorch (sequential)
You will need to import these libraries for pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
your data will have to be in tensor form for Pytorch this is the way I did it taking advantage of numpy datasets which were already split and standardized
XX_train = torch.from_numpy(X_train)
## targets = y_train.astype(int) ## cast to int
targets0 = np.eye(2)[y_train.astype(int)] ## one hot code target
yy_train = torch.from_numpy(np.eye(2)[y_train.astype(int)]).type(torch.FloatTensor)