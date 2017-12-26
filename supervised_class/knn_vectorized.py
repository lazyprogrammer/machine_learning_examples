# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# This is an example of a K-Nearest Neighbors classifier on MNIST data.
# We try k=1...5 to show how we might choose the best k.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from util import get_data
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        N = len(X)
        y = np.zeros(N)

        # returns distances in a matrix
        # of shape (N_test, N_train)
        distances = pairwise_distances(X, self.X)
        

        # now get the minimum k elements' indexes
        # https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
        idx = distances.argsort(axis=1)[:, :self.k]

        # now determine the winning votes
        # each row of idx contains indexes from 0..Ntrain
        # corresponding to the indexes of the closest samples
        # from the training set
        # NOTE: if you don't "believe" this works, test it
        # in your console with simpler arrays
        votes = self.y[idx]

        # now y contains the classes in each row
        # e.g.
        # sample 0 --> [class0, class1, class1, class0, ...]
        # unfortunately there's no good way to vectorize this
        # https://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays
        for i in range(N):
            y[i] = np.bincount(votes[i]).argmax()

        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    train_scores = []
    test_scores = []
    ks = (1,2,3,4,5)
    for k in ks:
        print("\nk =", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()

