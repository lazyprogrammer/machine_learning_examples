# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from util import getKaggleMNIST

import os
import sys
sys.path.append(os.path.abspath('..'))
from unsupervised_class.kmeans_mnist import purity
from unsupervised_class.gmm import gmm


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
    _, Rfull = gmm(X, 10, max_iter=30, smoothing=10e-1)
    print "full purity:", purity(Y, Rfull)
    _, Rreduced = gmm(Z, 10, max_iter=30, smoothing=10e-1)
    print "reduced purity:", purity(Y, Rreduced)

if __name__ == '__main__':
    main()