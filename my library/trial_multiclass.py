# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:14:58 2018

@author: e0046971
"""

from multiclass import multiclass

from multiclass_reg import multiclass_reg

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# calculating score 0-1
def score(Y,Ypred):
    return np.sum((Y==Ypred))/Y.shape[1]


np.random.seed(12)

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

ind=np.array([i for i in range(150)])
ind_train=np.random.choice(X.shape[0],120)
ind_test=np.delete(ind,ind_train)

X_train=X[ind_train,:].T
Y_train=Y[ind_train].reshape(1,Y[ind_train].shape[0])

X_test=X[ind_test,:].T
Y_test=Y[ind_test].reshape(1,Y[ind_test].shape[0])



softreg=multiclass_reg(X_train,Y_train,iterations=10000,learning_rate=0.01,lambd=0.01)
softreg.fit()

soft=multiclass(X_train,Y_train,iterations=10000,learning_rate=0.01)
soft.fit()


plt.figure()
plt.plot(softreg.error,label="with lambda")
plt.plot(soft.error,label="without lambda")
plt.title("Multiclass classification")
plt.legend()

Y_pred_reg=softreg.predict(X_test)
Y_pred_reg_train=softreg.predict(X_train)
print("Score in test set with lambda: " +str(score(Y_test,Y_pred_reg)))
print("Score in train set with lambda: " +str(score(Y_train,Y_pred_reg_train)))


Y_pred=soft.predict(X_test)
Y_pred_train=soft.predict(X_train)
print("Score in test set without lambda: " +str(score(Y_test,Y_pred)))
print("Score in train set without lambda: " +str(score(Y_train,Y_pred_train)))