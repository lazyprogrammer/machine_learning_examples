from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np

from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()


class SigmoidFeaturizer:
  def __init__(self, gamma=1.0, n_components=100, method='random'):
    self.M = n_components
    self.gamma = gamma
    assert(method in ('random', 'kmeans'))
    self.method = method

  def fit(self, X, Y=None):
    if self.method == 'random':
      N = len(X)
      idx = np.random.randint(N, size=self.M)
      self.samples = X[idx]
    elif self.method == 'kmeans':
      print("Fitting kmeans...")
      t0 = datetime.now()
      kmeans = KMeans(n_clusters=self.M)
      kmeans.fit(X)
      print("Finished fitting kmeans, duration:", datetime.now() - t0)
      self.samples = kmeans.cluster_centers_
    return self

  def transform(self, X):
    Z = self.gamma * X.dot(self.samples.T) # (Ntest x D) x (D x Nsamples) -> (Ntest x Nsamples)
    return np.tanh(Z)

  def fit_transform(self, X, Y=None):
    return self.fit(X, Y).transform(X)


# with SGD
pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('sigmoid', SigmoidFeaturizer(gamma=0.05, n_components=2000, method='random')),
  ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))
])

# with Linear SVC
# n_components = 3000
# pipeline = Pipeline([
#   ('scaler', StandardScaler()),
#   ('sigmoid', SigmoidFeaturizer(n_components=n_components)),
#   ('linear', LinearSVC())
# ])


t0 = datetime.now()
pipeline.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", pipeline.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", pipeline.score(Xtest, Ytest), "duration:", datetime.now() - t0)
