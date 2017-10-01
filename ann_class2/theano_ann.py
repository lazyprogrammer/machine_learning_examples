# For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from sklearn.utils import shuffle


def init_weight(M1, M2):
  return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)


class HiddenLayer(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

    def forward(self, X):
        if self.f == T.nnet.relu:
            return self.f(X.dot(self.W) + self.b, alpha=0.1)
        return self.f(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, activation=T.nnet.relu, learning_rate=1e-3, mu=0.0, reg=0, epochs=100, batch_sz=None, print_period=100, show_fig=True):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)

        # initialize hidden layers
        N, D = X.shape
        self.layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation)
            self.layers.append(h)
            M1 = M2
        
        # final layer
        K = len(set(Y))
        # print("K:", K)
        h = HiddenLayer(M1, K, T.nnet.softmax)
        self.layers.append(h)

        if batch_sz is None:
            batch_sz = N

        # collect params for later use
        self.params = []
        for h in self.layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]

        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.ivector('Y')
        p_y_given_x = self.forward(thX)

        rcost = reg*T.mean([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(p_y_given_x[T.arange(thY.shape[0]), thY])) #+ rcost
        prediction = T.argmax(p_y_given_x, axis=1)
        grads = T.grad(cost, self.params)

        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
        )

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
        )

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            if n_batches > 1:
              X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)
                if (j+1) % print_period == 0:
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c)
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        out = X
        for h in self.layers:
            out = h.forward(out)
        return out

    def score(self, X, Y):
        P = self.predict_op(X)
        return np.mean(Y == P)

    def predict(self, X):
        return self.predict_op(X)
