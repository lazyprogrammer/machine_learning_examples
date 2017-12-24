# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


class SimpleRNN:
    def __init__(self, M):
        self.M = M # hidden layer size

    def fit(self, X, Y, batch_sz=20, learning_rate=1.0, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1] # X is of size N x T(n) x D
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f = activation

        # initial weights
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        # make them theano shared
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X') # will represent multiple batches concatenated
        thY = T.ivector('Y')
        thStartPoints = T.ivector('start_points')

        XW = thX.dot(self.Wx)

        # startPoints will contain 1 where a sequence starts and 0 otherwise
        # Ex. if I have 3 sequences: [[1,2,3], [4,5], [6,7,8]]
        # Then I will concatenate these into one X: [1,2,3,4,5,6,7,8]
        # And startPoints will be [1,0,0,1,0,1,0,0]

        # One possible solution: loop through index
        # def recurrence(t, h_t1, XW, h0, startPoints):
        #     # returns h(t)

        #     # if at a boundary, state should be h0
        #     h_t = T.switch(
        #         T.eq(startPoints[t], 1),
        #         self.f(XW[t] + h0.dot(self.Wh) + self.bh),
        #         self.f(XW[t] + h_t1.dot(self.Wh) + self.bh)
        #     )
        #     return h_t

        # h, _ = theano.scan(
        #     fn=recurrence,
        #     outputs_info=[self.h0],
        #     sequences=T.arange(XW.shape[0]),
        #     non_sequences=[XW, self.h0, thStartPoints],
        #     n_steps=XW.shape[0],
        # )

        # other solution - loop through all sequences simultaneously
        def recurrence(xw_t, is_start, h_t1, h0):
            # if at a boundary, state should be h0
            h_t = T.switch(
                T.eq(is_start, 1),
                self.f(xw_t + h0.dot(self.Wh) + self.bh),
                self.f(xw_t + h_t1.dot(self.Wh) + self.bh)
            )
            return h_t

        h, _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0],
            sequences=[XW, thStartPoints],
            non_sequences=[self.h0],
            n_steps=XW.shape[0],
        )

        # h is of shape (T*batch_sz, M)
        py_x = T.nnet.softmax(h.dot(self.Wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        # self.predict_op = theano.function(inputs=[thX, thStartPoints], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY, thStartPoints],
            outputs=[cost, prediction, py_x],
            updates=updates
        )

        costs = []
        n_batches = N // batch_sz
        sequenceLength = X.shape[1]

        # if each sequence was of variable length, we would need to
        # initialize this inside the loop for every new batch
        startPoints = np.zeros(sequenceLength*batch_sz, dtype=np.int32)
        for b in range(batch_sz):
            startPoints[b*sequenceLength] = 1
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j+1)*batch_sz].reshape(sequenceLength*batch_sz, D)
                Ybatch = Y[j*batch_sz:(j+1)*batch_sz].reshape(sequenceLength*batch_sz).astype(np.int32)
                c, p, rout = self.train_op(Xbatch, Ybatch, startPoints)
                # print "p:", p
                cost += c
                # P = p.reshape(batch_sz, sequenceLength)
                for b in range(batch_sz):
                    idx = sequenceLength*(b + 1) - 1
                    if p[idx] == Ybatch[idx]:
                        n_correct += 1
                    # else:
                        # print "pred:", p[idx], "actual:", Ybatch[idx]
            if i % 10 == 0:
                print("shape y:", rout.shape)
                print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
            if n_correct == N:
                print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                break
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()



def parity(B=12, learning_rate=1e-3, epochs=3000):
    X, Y = all_parity_pairs_with_sequence_labels(B)

    rnn = SimpleRNN(4)
    rnn.fit(X, Y,
        batch_sz=10,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=T.nnet.sigmoid,
        show_fig=False
    )


if __name__ == '__main__':
    parity()

