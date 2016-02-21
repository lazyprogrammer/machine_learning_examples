# assume a 3-layer net (1 hidden)
# TODO: don't use this! if it was committed that was by mistake, it's not even done!

import numpy as np
import pandas as pd


Xtest = pd.read_csv("mnist_csv/Xtest.txt", header=None)
Xtrain = pd.read_csv("mnist_csv/Xtrain.txt", header=None)
Ytest = pd.read_csv("mnist_csv/label_test.txt", header=None)
Ytrain = pd.read_csv("mnist_csv/label_train.txt", header=None)


class ANN(object):
    def __init__(self, n_hidden, epochs=1000, learning_rate=0.00001, regularization=0.1):
        self.n_hidden = n_hidden


    def feedforward(self, X):
        # expects X to be 2-D
        z = X.dot(self.V)
        hidden = 1 / (1 + np.exp(-z))
        y = hidden.dot(self.W)
        # return 1 / (1 + np.exp(-y))
        numerators = np.exp(y)
        return numerators


    def softmax(self, X):
        # slow
        numerators = self.feedforward(X)
        sums = numerators.sum(axis=1)
        output = np.zeros(numerators.shape)
        for n in xrange(numerators.shape[0]):
            output[n,:] = numerators[n,:] / sums[n]
        return output


    def train(self, X, Y):
        N, D = X.shape
        K = len(set(Y))
        self.V = np.random.randn(D, n_hidden)
        self.W = np.random.randn(n_hidden, K)

        for epoch in xrange(epochs):
            prediction = self.softmax(X)
            self.W += learning_rate * (Y - prediction).dot(X)


    def predict(self, X):
        P = self.feedforward(X)
        return P.argmax(axis=1)






