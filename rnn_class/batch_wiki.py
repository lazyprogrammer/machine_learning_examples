# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import json

from datetime import datetime
from sklearn.utils import shuffle
from batch_units import GRU, LSTM
from util import init_weight, get_wikipedia_data
from brown import get_sentences_with_word2idx_limit_vocab


class RNN:
    def __init__(self, D, hidden_layer_sizes, V):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D
        self.V = V

    def fit(self, X, learning_rate=1e-4, mu=0.99, epochs=10, batch_sz=100, show_fig=True, activation=T.nnet.relu, RecurrentUnit=LSTM):
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

        Wo = init_weight(Mi, V)
        bo = np.zeros(V)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        thX = T.ivector('X') # will represent multiple batches concatenated
        thY = T.ivector('Y') # represents next word
        thStartPoints = T.ivector('start_points')

        Z = self.We[thX]
        for ru in self.hidden_layers:
            Z = ru.output(Z, thStartPoints)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)
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
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        n_batches = N // batch_sz
        for i in range(epochs):
            t0 = datetime.now()
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0

            for j in range(n_batches):
                # construct input sequence and output sequence as
                # concatenatation of multiple input sequences and output sequences
                # input X should be a list of 2-D arrays or one 3-D array
                # N x T(n) x D - batch size x sequence length x num features
                # sequence length can be variable
                sequenceLengths = []
                input_sequence = []
                output_sequence = []
                for k in range(j*batch_sz, (j+1)*batch_sz):
                    # don't always add the end token
                    if np.random.random() < 0.01 or len(X[k]) <= 1:
                        input_sequence += [0] + X[k]
                        output_sequence += X[k] + [1]
                        sequenceLengths.append(len(X[k]) + 1)
                    else:
                        input_sequence += [0] + X[k][:-1]
                        output_sequence += X[k]
                        sequenceLengths.append(len(X[k]))
                n_total += len(output_sequence)

                startPoints = np.zeros(len(output_sequence), dtype=np.int32)
                last = 0
                for length in sequenceLengths:
                  startPoints[last] = 1
                  last += length

                c, p = self.train_op(input_sequence, output_sequence, startPoints)
                cost += c
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1
                if j % 1 == 0:
                    sys.stdout.write("j/n_batches: %d/%d correct rate so far: %f\r" % (j, n_batches, float(n_correct)/n_total))
                    sys.stdout.flush()
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total), "time for epoch:", (datetime.now() - t0))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()



def train_wikipedia(we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json', RecurrentUnit=GRU):
    # there are 32 files
    ### note: you can pick between Wikipedia data and Brown corpus
    ###       just comment one out, and uncomment the other!
    # sentences, word2idx = get_wikipedia_data(n_files=100, n_vocab=2000)
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab()
    print("finished retrieving data")
    print("vocab size:", len(word2idx), "number of sentences:", len(sentences))
    rnn = RNN(30, [30], len(word2idx))
    rnn.fit(sentences, learning_rate=2*1e-4, epochs=10, show_fig=True, activation=T.nnet.relu)

    np.save(we_file, rnn.We.get_value())
    with open(w2i_file, 'w') as f:
        json.dump(word2idx, f)



def find_analogies(w1, w2, w3, we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json'):
    We = np.load(we_file)
    with open(w2i_file) as f:
        word2idx = json.load(f)

    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    def dist1(a, b):
        return np.linalg.norm(a - b)
    def dist2(a, b):
        return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
        min_dist = float('inf')
        best_word = ''
        for word, idx in iteritems(word2idx):
            if word not in (w1, w2, w3):
                v1 = We[idx]
                d = dist(v0, v1)
                if d < min_dist:
                    min_dist = d
                    best_word = word
        print("closest match by", name, "distance:", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)



if __name__ == '__main__':
    if not os.path.exists('working_files'):
        os.mkdir('working_files')
    we = 'working_files/batch_gru_word_embeddings.npy'
    w2i = 'working_files/batch_wikipedia_word2idx.json'
    train_wikipedia(we, w2i, RecurrentUnit=LSTM)
    find_analogies('king', 'man', 'woman', we, w2i)
    find_analogies('france', 'paris', 'london', we, w2i)
    find_analogies('france', 'paris', 'rome', we, w2i)
    find_analogies('paris', 'france', 'italy', we, w2i)

