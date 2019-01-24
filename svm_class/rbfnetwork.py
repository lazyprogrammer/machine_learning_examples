# https://deeplearningcourses.com/c/support-vector-machines-in-python
# https://www.udemy.com/support-vector-machines-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem

# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# linear SGD classifier
# pipeline = Pipeline([('linear', SGDClassifier(max_iter=1e6, tol=1e-5))])

# linear SVC - a bit faster than SVC with linear kernel
# pipeline = Pipeline([('linear', LinearSVC())])

# one RBFSampler with linear SGD classifier
# pipeline = Pipeline([
#   ('rbf', RBFSampler(gamma=0.01, n_components=1000)),
#   ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))])

# multiple RBFSamplers
# n_components = 2000
# featurizer = FeatureUnion([
#   ("rbf1", RBFSampler(gamma=0.01, n_components=n_components)),
#   ("rbf2", RBFSampler(gamma=0.005, n_components=n_components)),
#   ("rbf3", RBFSampler(gamma=0.001, n_components=n_components)),
#   ])
# pipeline = Pipeline([('rbf', featurizer), ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))])

# Nystroem approximation
# pipeline = Pipeline([
#   ('rbf', Nystroem(gamma=0.05, n_components=1000)),
#   ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))])

# multiple Nystroem
n_components = 1000
featurizer = FeatureUnion([
  ("rbf0", Nystroem(gamma=0.05, n_components=n_components)),
  ("rbf1", Nystroem(gamma=0.01, n_components=n_components)),
  ("rbf2", Nystroem(gamma=0.005, n_components=n_components)),
  ("rbf3", Nystroem(gamma=0.001, n_components=n_components)),
  ])
pipeline = Pipeline([('rbf', featurizer), ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))])


t0 = datetime.now()
pipeline.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", pipeline.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", pipeline.score(Xtest, Ytest), "duration:", datetime.now() - t0)
