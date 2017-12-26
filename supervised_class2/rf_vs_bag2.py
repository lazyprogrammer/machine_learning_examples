# https://deeplearningcourses.com/c/machine-learning-in-python-random-forest-adaboost
# https://www.udemy.com/machine-learning-in-python-random-forest-adaboost
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, RandomForestClassifier, BaggingClassifier
from util import BaggedTreeRegressor, BaggedTreeClassifier

# make simple regression data
# N = 15
# D = 100
# X = (np.random.random((N, D)) - 0.5)*10
# Y = X.sum(axis=1)**2 + 0.5*np.random.randn(N)
# Ntrain = N/2
# Xtrain = X[:Ntrain]
# Ytrain = Y[:Ntrain]
# Xtest = X[Ntrain:]
# Ytest = Y[Ntrain:]

from rf_classification import get_data
X, Y = get_data()
Ntrain = int(0.8*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# from rf_regression import get_data
# Xtrain, Ytrain, Xtest, Ytest = get_data()

class NotAsRandomForest:
  def __init__(self, n_estimators):
    self.B = n_estimators

  def fit(self, X, Y, M=None):
    N, D = X.shape
    if M is None:
      M = int(np.sqrt(D))

    self.models = []
    self.features = []
    for b in range(self.B):
      tree = DecisionTreeClassifier()

      # sample features
      features = np.random.choice(D, size=M, replace=False)

      # sample training samples
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]

      tree.fit(Xb[:, features], Yb)
      self.features.append(features)
      self.models.append(tree)

  def predict(self, X):
    N = len(X)
    P = np.zeros(N)
    for features, tree in zip(self.features, self.models):
      P += tree.predict(X[:, features])
    return np.round(P / self.B)

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)


T = 500
test_error_prf = np.empty(T)
test_error_rf = np.empty(T)
test_error_bag = np.empty(T)
for num_trees in range(T):
  if num_trees == 0:
    test_error_prf[num_trees] = None
    test_error_rf[num_trees] = None
    test_error_bag[num_trees] = None
  else:
    rf = RandomForestClassifier(n_estimators=num_trees)
    rf.fit(Xtrain, Ytrain)
    test_error_rf[num_trees] = rf.score(Xtest, Ytest)

    bg = BaggedTreeClassifier(n_estimators=num_trees)
    bg.fit(Xtrain, Ytrain)
    test_error_bag[num_trees] = bg.score(Xtest, Ytest)

    prf = NotAsRandomForest(n_estimators=num_trees)
    prf.fit(Xtrain, Ytrain)
    test_error_prf[num_trees] = prf.score(Xtest, Ytest)

  if num_trees % 10 == 0:
    print("num_trees:", num_trees)

plt.plot(test_error_rf, label='rf')
plt.plot(test_error_prf, label='pseudo rf')
plt.plot(test_error_bag, label='bag')
plt.legend()
plt.show()
