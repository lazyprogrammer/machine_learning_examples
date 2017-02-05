# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
import numpy as np
import matplotlib.pyplot as plt
from util import get_data as get_mnist
from datetime import datetime


def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2))*2 - 1
    Y = np.sign(X.dot(w) + b)
    return X, Y


def get_simple_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    return X, Y


class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        # solution
        # self.w = np.array([-0.5, 0.5])
        # self.b = 0.1

        # initialize random weights
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        costs = []
        for epoch in xrange(epochs):
            # determine which samples are misclassified, if any
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                # we are done!
                break

            # choose a random incorrect sample
            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            # cost is incorrect rate
            c = len(incorrect) / float(N)
            costs.append(c)
        print "final w:", self.w, "final b:", self.b, "epochs:", (epoch+1), "/", epochs
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    # linearly separable data
    X, Y = get_data()
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print "Training time:", (datetime.now() - t0)

    t0 = datetime.now()
    print "Train accuracy:", model.score(Xtrain, Ytrain)
    print "Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain)

    t0 = datetime.now()
    print "Test accuracy:", model.score(Xtest, Ytest)
    print "Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest)


    # mnist
    X, Y = get_mnist()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X, Y, learning_rate=10e-3)
    print "MNIST train accuracy:", model.score(X, Y)


    # xor data
    print ""
    print "XOR results:"
    X, Y = get_simple_xor()
    model.fit(X, Y)
    print "XOR accuracy:", model.score(X, Y)

