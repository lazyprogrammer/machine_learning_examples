# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# This is an example of a Naive Bayes classifier on MNIST data.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from util import get_data
from datetime import datetime

class MultinomialNB(object):
    def fit(self, X, Y, smoothing=1.0):
        # one-hot encode Y
        K = len(set(Y)) # number of classes
        N = len(Y) # number of samples
        labels = Y
        Y = np.zeros((N, K))
        Y[np.arange(N), labels] = 1

        # D x K matrix of feature counts
        # feature_counts[d,k] = count of feature d in class k
        feature_counts = X.T.dot(Y) + smoothing
        class_counts = Y.sum(axis=0)

        self.weights = np.log(feature_counts) - np.log(feature_counts.sum(axis=0))
        self.priors = np.log(class_counts) - np.log(class_counts.sum())

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        P = X.dot(self.weights) + self.priors
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X, Y = get_data(10000)
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = MultinomialNB()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
