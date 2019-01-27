# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, TruncatedSVD
from util import getKaggleMNIST


X, Y, _, _ = getKaggleMNIST()
m = X.mean(axis=0)
s = X.std(axis=0)
np.place(s, s == 0, 1)
X = (X - m) / s

pca = PCA()
svd = TruncatedSVD()

Z1 = pca.fit_transform(X)
Z2 = svd.fit_transform(X)

plt.subplot(1,2,1)
plt.scatter(Z1[:,0], Z1[:,1], c=Y)
plt.subplot(1,2,2)
plt.scatter(Z2[:,0], Z2[:,1], c=Y)
plt.show()
