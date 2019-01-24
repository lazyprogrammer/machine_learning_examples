# https://deeplearningcourses.com/c/support-vector-machines-in-python
# https://www.udemy.com/support-vector-machines-in-python
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
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from scipy import stats
from sklearn.linear_model import LogisticRegression


class SigmoidFeaturizer:
  def __init__(self, gamma=1.0, n_components=100, method='random'):
    self.M = n_components
    self.gamma = gamma
    assert(method in ('normal', 'random', 'kmeans', 'gmm'))
    self.method = method

  def _subsample_data(self, X, Y, n=10000):
    if Y is not None:
      X, Y = shuffle(X, Y)
      return X[:n], Y[:n]
    else:
      X = shuffle(X)
      return X[:n]

  def fit(self, X, Y=None):
    if self.method == 'random':
      N = len(X)
      idx = np.random.randint(N, size=self.M)
      self.samples = X[idx]
    elif self.method == 'normal':
      # just sample from N(0,1)
      D = X.shape[1]
      self.samples = np.random.randn(self.M, D) / np.sqrt(D)
    elif self.method == 'kmeans':
      X, Y = self._subsample_data(X, Y)

      print("Fitting kmeans...")
      t0 = datetime.now()
      kmeans = KMeans(n_clusters=len(set(Y)))
      kmeans.fit(X)
      print("Finished fitting kmeans, duration:", datetime.now() - t0)

      # calculate the most ambiguous points
      # we will do this by finding the distance between each point
      # and all cluster centers
      # and return which points have the smallest variance
      dists = kmeans.transform(X) # returns an N x K matrix
      variances = dists.var(axis=1)
      idx = np.argsort(variances) # smallest to largest
      idx = idx[:self.M]
      self.samples = X[idx]
    elif self.method == 'gmm':
      X, Y = self._subsample_data(X, Y)

      print("Fitting GMM")
      t0 = datetime.now()
      gmm = GaussianMixture(
        n_components=len(set(Y)),
        covariance_type='spherical',
        reg_covar=1e-6)
      gmm.fit(X)
      print("Finished fitting GMM, duration:", datetime.now() - t0)

      # calculate the most ambiguous points
      probs = gmm.predict_proba(X)
      ent = stats.entropy(probs.T) # N-length vector of entropies
      idx = np.argsort(-ent) # negate since we want biggest first
      idx = idx[:self.M]
      self.samples = X[idx]
    return self

  def transform(self, X):
    Z = X.dot(self.samples.T) # (Ntest x D) x (D x Nsamples) -> (Ntest x Nsamples)
    return np.tanh(self.gamma * Z)
    # return self.gamma * Z * (Z > 0)

  def fit_transform(self, X, Y=None):
    return self.fit(X, Y).transform(X)


# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# with SGD
pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('sigmoid', SigmoidFeaturizer(gamma=0.05, n_components=2000, method='normal')),
  # ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))
  ('linear', LogisticRegression()) # takes longer
])

# with Linear SVC
# n_components = 3000
# pipeline = Pipeline([
#   ('scaler', StandardScaler()),
#   ('sigmoid', SigmoidFeaturizer(n_components=n_components)),
#   ('linear', LinearSVC())
# ])

# let's do some cross-validation instead, why not
X = np.vstack((Xtrain, Xtest))
Y = np.concatenate((Ytrain, Ytest))
scores = cross_val_score(pipeline, X, Y, cv=5)
print(scores)
print("avg:", np.mean(scores))

# t0 = datetime.now()
# pipeline.fit(Xtrain, Ytrain)
# print("train duration:", datetime.now() - t0)
# t0 = datetime.now()
# print("train score:", pipeline.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
# t0 = datetime.now()
# print("test score:", pipeline.score(Xtest, Ytest), "duration:", datetime.now() - t0)
