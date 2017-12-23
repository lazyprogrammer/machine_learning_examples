# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost


class SimpleRNN:
    def __init__(self, D, M, V):
        self.D = D # dimensionality of word embedding
        self.M = M # hidden layer size
        self.V = V # vocabulary size

    def fit(self, X, learning_rate=10., mu=0.9, reg=0., activation=T.tanh, epochs=500, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        # initial weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        # z  = np.ones(M)
        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        bz  = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)

        lr = T.scalar('lr')

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = []
        for p, dp, g in zip(self.params, dparams, grads):
            new_dp = mu*dp - lr*g
            updates.append((dp, new_dp))

            new_p = p + new_dp
            updates.append((p, new_p))

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY, lr],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                if np.random.random() < 0.1:
                    input_sequence = [0] + X[j]
                    output_sequence = X[j] + [1]
                else:
                    input_sequence = [0] + X[j][:-1]
                    output_sequence = X[j]
                n_total += len(output_sequence)

                # we set 0 to start and 1 to end
                c, p = self.train_op(input_sequence, output_sequence, learning_rate)
                # print "p:", p
                cost += c
                # print "j:", j, "c:", c/len(X[j]+1)
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total))
            if (i + 1) % 500 == 0:
                learning_rate /= 2
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()
    
    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation):
        # TODO: would prefer to save activation to file too
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wxz = npz['arr_5']
        Whz = npz['arr_6']
        bz = npz['arr_7']
        Wo = npz['arr_8']
        bo = npz['arr_9']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation):
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bz)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wxz, self.Whz, self.bz, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            hhat_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
            h_t = (1 - z_t) * h_t1 + z_t * hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[py_x, prediction],
            allow_input_downcast=True,
        )
        return thX, thY, py_x, prediction


    def generate(self, word2idx):
        # convert word2idx -> idx2word
        idx2word = {v:k for k,v in iteritems(word2idx)}
        V = len(word2idx)

        # generate 4 lines at a time
        n_lines = 0

        # why? because using the START symbol will always yield the same first word!
        X = [ 0 ]
        while n_lines < 4:
            # print "X:", X
            PY_X, _ = self.predict_op(X)
            PY_X = PY_X[-1].flatten()
            P = [ np.random.choice(V, p=PY_X)]
            X = np.concatenate([X, P]) # append to the sequence
            # print "P.shape:", P.shape, "P:", P
            P = P[-1] # just grab the most recent prediction
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word[P]
                print(word, end=" ")
            elif P == 1:
                # end token
                n_lines += 1
                X = [0]
                print('')


def train_poetry():
    # students: tanh didn't work but you should try it
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(50, 50, len(word2idx))
    rnn.fit(sentences, learning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=2000)
    rnn.save('RRNN_D50_M50_epochs2000_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RRNN_D50_M50_epochs2000_relu.npz', T.nnet.relu)
    rnn.generate(word2idx)


if __name__ == '__main__':
    train_poetry()
    generate_poetry()

