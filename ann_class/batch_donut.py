# revisiting the XOR and donut problems to show how features
# can be learned automatically using neural networks.
# since full training didn't work so well with the donut
# problem, let's try learning from randomly selected batches.
# we can consistently get up to the high 90s with this method.
#
# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-deep-learning-in-python
# https://www.udemy.com/data-science-deep-learning-in-python

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# for binary classification! no softmax here

def forward(X, W1, b1, W2, b2):
    # assume we will use tanh() on hidden
    # and softmax on output
    Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))
    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z


def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)


def derivative_w2(Z, T, Y):
    # Z is (N, M)
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()


def derivative_w1(X, Z, T, Y, W2):
    front = np.outer(T-Y, W2) * Z * (1 - Z)
    return front.T.dot(X).T


def derivative_b1(Z, T, Y, W2):
    front = np.outer(T-Y, W2) * Z * (1 - Z)
    return front.sum(axis=0)


def cost(T, Y):
    # tot = 0
    # for n in xrange(len(T)):
    #     if T[n] == 1:
    #         tot += np.log(Y[n])
    #     else:
    #         tot += np.log(1 - Y[n])
    # return tot
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

def test_donut():
    # donut example
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N/2) + [1]*(N/2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = [] # keep track of likelihoods
    learning_rate = 0.0001
    regularization = 0.1


    # batch version
    batch_size = 100
    for i in xrange(150000):
        tmpX, tmpY = shuffle(X, Y)

        tmpX = tmpX[:batch_size]
        tmpY = tmpY[:batch_size]
        pY, Z = forward(tmpX, W1, b1, W2, b2)
        ll = cost(tmpY, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, tmpY, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(tmpY, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(tmpX, Z, tmpY, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, tmpY, pY, W2) - regularization * b1)
        if i % 100 == 0:
            print "ll:", ll, "classification rate:", 1 - er

    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    test_donut()