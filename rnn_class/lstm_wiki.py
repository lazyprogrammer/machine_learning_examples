# OLD - use wiki.py instead
import json
import numpy as np
import theano
import theano.tensor as T

from util import init_weight, get_wikipedia_data
from gru_wiki import RNN, find_analogies

class LSTM:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi  = np.zeros(Mo)
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf  = np.zeros(Mo)
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc  = np.zeros(Mo)
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo  = np.zeros(Mo)
        c0  = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi  = theano.shared(bi)
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf  = theano.shared(bf)
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc  = theano.shared(bc)
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo  = theano.shared(bo)
        self.c0  = theano.shared(c0)
        self.h0  = theano.shared(h0)
        self.params = [
            self.Wxi,
            self.Whi,
            self.Wci,
            self.bi,
            self.Wxf,
            self.Whf,
            self.Wcf,
            self.bf,
            self.Wxc,
            self.Whc,
            self.bc,
            self.Wxo,
            self.Who,
            self.Wco,
            self.bo,
            self.c0,
            self.h0,
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        # input X should be a matrix (2-D)
        # rows index time
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0, self.c0],
            n_steps=x.shape[0],
        )
        return h


def train_wikipedia(we_file='lstm_word_embeddings.npy', w2i_file='lstm_wikipedia_word2idx.json'):
    # there are 32 files
    sentences, word2idx = get_wikipedia_data(n_files=100, n_vocab=2000)
    print "finished retrieving data"
    print "vocab size:", len(word2idx), "number of sentences:", len(sentences)
    rnn = RNN(50, [50], len(word2idx))
    # todo: next try increas LR
    rnn.fit(sentences, learning_rate=10e-6, epochs=10, show_fig=True, activation=T.nnet.relu, RecurrentUnit=LSTM, normalize=False)

    np.save(we_file, rnn.We.get_value())
    with open(w2i_file, 'w') as f:
        json.dump(word2idx, f)


if __name__ == '__main__':
    train_wikipedia()
    find_analogies('king', 'man', 'woman', 'lstm_word_embeddings.npy', 'lstm_wikipedia_word2idx.json')
    find_analogies('france', 'paris', 'london', 'lstm_word_embeddings.npy', 'lstm_wikipedia_word2idx.json')
    find_analogies('france', 'paris', 'rome', 'lstm_word_embeddings.npy', 'lstm_wikipedia_word2idx.json')
    find_analogies('paris', 'france', 'italy', 'lstm_word_embeddings.npy', 'lstm_wikipedia_word2idx.json')

