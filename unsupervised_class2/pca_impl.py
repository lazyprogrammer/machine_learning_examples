# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from util import getKaggleMNIST

# get the data
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# decompose covariance
covX = np.cov(Xtrain.T)
lambdas, Q = np.linalg.eigh(covX)


# lambdas are sorted from smallest --> largest
# some may be slightly negative due to precision
idx = np.argsort(-lambdas)
lambdas = lambdas[idx] # sort in proper order
lambdas = np.maximum(lambdas, 0) # get rid of negatives
Q = Q[:,idx]


# plot the first 2 columns of Z
Z = Xtrain.dot(Q)
plt.scatter(Z[:,0], Z[:,1], s=100, c=Ytrain, alpha=0.3)
plt.show()


# plot variances
plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

# cumulative variance
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()