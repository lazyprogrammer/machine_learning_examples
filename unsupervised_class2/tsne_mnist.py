# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from util import getKaggleMNIST

import os
import sys
sys.path.append(os.path.abspath('..'))
from unsupervised_class.kmeans_mnist import purity
from sklearn.mixture import GaussianMixture


def main():
    Xtrain, Ytrain, _, _ = getKaggleMNIST()

    sample_size = 1000
    X = Xtrain[:sample_size]
    Y = Ytrain[:sample_size]

    tsne = TSNE()
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # purity measure from unsupervised machine learning pt 1
    # maximum purity is 1, higher is better
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X)
    Rfull = gmm.predict_proba(X)
    print("Rfull.shape:", Rfull.shape)
    print("full purity:", purity(Y, Rfull))

    # now try the same thing on the reduced data
    gmm.fit(Z)
    Rreduced = gmm.predict_proba(Z)
    print("reduced purity:", purity(Y, Rreduced))

if __name__ == '__main__':
    main()