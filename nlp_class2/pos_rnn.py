# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.gru import GRU
from pos_baseline import get_data
from sklearn.utils import shuffle
from util import init_weight
from datetime import datetime
from sklearn.metrics import f1_score


class RNN:
    def __init__(self, D, hidden_layer_sizes, V, K):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D
        self.V = V
        self.K = K

    def fit(self, X, Y, learning_rate=1e-4, mu=0.99, epochs=30, show_fig=True, activation=T.nnet.relu, RecurrentUnit=GRU, normalize=False):
        D = self.D
        V = self.V
        N = len(X)

        We = init_weight(V, D)
        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        Wo = init_weight(Mi, self.K)
        bo = np.zeros(self.K)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        thX = T.ivector('X')
        thY = T.ivector('Y')

        Z = self.We[thX]
        for ru in self.hidden_layers:
            Z = ru.output(Z)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

        testf = theano.function(
            inputs=[thX],
            outputs=py_x,
        )
        testout = testf(X[0])
        print("py_x.shape:", testout.shape)

        prediction = T.argmax(py_x, axis=1)
        
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        dWe = theano.shared(self.We.get_value()*0)
        gWe = T.grad(cost, self.We)
        dWe_update = mu*dWe - learning_rate*gWe
        We_update = self.We + dWe_update
        if normalize:
            We_update /= We_update.norm(2)

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] + [
            (self.We, We_update), (dWe, dWe_update)
        ]

        self.cost_predict_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        sequence_indexes = range(N)
        n_total = sum(len(y) for y in Y)
        for i in range(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            cost = 0
            it = 0
            for j in sequence_indexes:
                c, p = self.train_op(X[j], Y[j])
                cost += c
                n_correct += np.sum(p == Y[j])
                it += 1
                if it % 200 == 0:
                    sys.stdout.write(
                        "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
                        (it, N, float(n_correct)/n_total, cost)
                    )
                    sys.stdout.flush()
            print(
                "i:", i, "cost:", cost,
                "correct rate:", (float(n_correct)/n_total),
                "time for epoch:", (datetime.now() - t0)
            )
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def score(self, X, Y):
        n_total = sum(len(y) for y in Y)
        n_correct = 0
        for x, y in zip(X, Y):
            _, p = self.cost_predict_op(x, y)
            n_correct += np.sum(p == y)
        return float(n_correct) / n_total

    def f1_score(self, X, Y):
        P = []
        for x, y in zip(X, Y):
            _, p = self.cost_predict_op(x, y)
            P.append(p)
        Y = np.concatenate(Y)
        P = np.concatenate(P)
        return f1_score(Y, P, average=None).mean()


def flatten(l):
    return [item for sublist in l for item in sublist]


def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
    V = len(word2idx) + 1
    K = len(set(flatten(Ytrain)) | set(flatten(Ytest)))
    rnn = RNN(10, [10], V, K)
    rnn.fit(Xtrain, Ytrain)
    print("train score:", rnn.score(Xtrain, Ytrain))
    print("test score:", rnn.score(Xtest, Ytest))
    print("train f1:", rnn.f1_score(Xtrain, Ytrain))
    print("test f1:", rnn.f1_score(Xtest, Ytest))
    

if __name__ == '__main__':
    main()
