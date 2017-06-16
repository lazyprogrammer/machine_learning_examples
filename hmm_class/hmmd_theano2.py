# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Discrete Hidden Markov Model (HMM) in Theano using gradient descent.

# This script differs from hmmd_theano.py in the following way:
# Instead of re-normalizing the parameters at each iteration,
# we instead make the parameters free to vary between -inf to +inf.
# We then use softmax to ensure the probabilities are positive and sum to 1.


from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


class HMM:
    def __init__(self, M):
        self.M = M # number of hidden states
    
    def fit(self, X, learning_rate=0.001, max_iter=10, V=None, print_period=1):
        # train the HMM model using stochastic gradient descent
        # print "X to train:", X

        # determine V, the vocabulary size
        # assume observables are already integers from 0..V-1
        # X is a jagged array of observed sequences
        if V is None:
            V = max(max(x) for x in X) + 1
        N = len(X)
        print("number of train samples:", N)

        preSoftmaxPi0 = np.zeros(self.M) # initial state distribution
        preSoftmaxA0 = np.random.randn(self.M, self.M) # state transition matrix
        preSoftmaxB0 = np.random.randn(self.M, V) # output distribution

        thx, cost = self.set(preSoftmaxPi0, preSoftmaxA0, preSoftmaxB0)

        pi_update = self.preSoftmaxPi - learning_rate*T.grad(cost, self.preSoftmaxPi)
        A_update = self.preSoftmaxA - learning_rate*T.grad(cost, self.preSoftmaxA)
        B_update = self.preSoftmaxB - learning_rate*T.grad(cost, self.preSoftmaxB)

        updates = [
            (self.preSoftmaxPi, pi_update),
            (self.preSoftmaxA, A_update),
            (self.preSoftmaxB, B_update),
        ]

        train_op = theano.function(
            inputs=[thx],
            updates=updates,
            allow_input_downcast=True,
        )

        costs = []
        for it in range(max_iter):
            if it % print_period == 0:
                print("it:", it)
            
            for n in range(N):
                # this would of course be much faster if we didn't do this on
                # every iteration of the loop
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                train_op(X[n])

        # print "A:", self.A.get_value()
        # print "B:", self.B.get_value()
        # print "pi:", self.pi.get_value()
        plt.plot(costs)
        plt.show()

    def get_cost(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        # print "getting cost for:", x
        return self.cost_op(x)

    def log_likelihood(self, x):
        return -self.cost_op(x)

    def get_cost_multi(self, X):
        P = np.random.random(len(X))
        return np.array([self.get_cost(x) for x, p in zip(X, P)])

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
        self.preSoftmaxPi = theano.shared(preSoftmaxPi)
        self.preSoftmaxA = theano.shared(preSoftmaxA)
        self.preSoftmaxB = theano.shared(preSoftmaxB)

        pi = T.nnet.softmax(self.preSoftmaxPi).flatten()
        # softmax returns 1xD if input is a 1-D array of size D
        A = T.nnet.softmax(self.preSoftmaxA)
        B = T.nnet.softmax(self.preSoftmaxB)

        # define cost
        thx = T.ivector('thx')
        def recurrence(t, old_a, x):
            a = old_a.dot(A) * B[:, x[t]]
            s = a.sum()
            return (a / s), s

        [alpha, scale], _ = theano.scan(
            fn=recurrence,
            sequences=T.arange(1, thx.shape[0]),
            outputs_info=[pi*B[:,thx[0]], None],
            n_steps=thx.shape[0]-1,
            non_sequences=thx
        )

        cost = -T.log(scale).sum()
        self.cost_op = theano.function(
            inputs=[thx],
            outputs=cost,
            allow_input_downcast=True,
        )
        return thx, cost


def fit_coin():
    X = []
    for line in open('coin_data.txt'):
        # 1 for H, 0 for T
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.get_cost_multi(X).sum()
    print("LL with fitted params:", L)

    # try true values
    # remember these must be in their "pre-softmax" forms
    pi = np.log( np.array([0.5, 0.5]) )
    A = np.log( np.array([[0.1, 0.9], [0.8, 0.2]]) )
    B = np.log( np.array([[0.6, 0.4], [0.3, 0.7]]) )
    hmm.set(pi, A, B)
    L = hmm.get_cost_multi(X).sum()
    print("LL with true params:", L)


if __name__ == '__main__':
    fit_coin()
