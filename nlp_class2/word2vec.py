import re
import nltk
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle

import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data

# TODO: try with wikipedia data, nvocab = 2000
# TODO: see if theano can update a SUB-matrix

stopwords = set(w.rstrip() for w in open('../nlp_class/stopwords.txt'))
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', 'this'})

wordnet_lemmatizer = WordNetLemmatizer()

pattern = re.compile('[\W_]+ ')

def my_tokenizer(s):
    s = s.strip().lower() # downcase
    s = pattern.sub('', s)
    # tokens = nltk.tokenize.word_tokenize(s) # too slow!
    tokens = s.split()
    tokens = [t for t in tokens if len(t) > 2 and t not in stopwords and not any(c.isdigit() for c in t)]
    # tokens = [t for t in tokens if t not in stopwords]
    # tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_weights(shape):
    return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))


class Model(object):
    def __init__(self, D, V, context_sz):
        self.D = D # embedding dimension
        self.V = V # vocab size
        self.context_sz = context_sz

    # naive implementation
    # regular backprop with full matrices
    def old_fit(self, sentences, context_size=5, learning_rate=1.0, mu=0.99, reg=0.1, epochs=8, batch_sz=20):
        # sentences can be an iterator of sentences

        # recast
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        reg = np.float32(reg)

        # first determine the vocabulary size
        word_freq = {}
        word_count = 0
        word2idx = {}
        idx2word = []
        idx = 0
        # pairs = []
        mid = context_size / 2
        X = []
        Y = []
        num_sentences = 0
        for sentence in sentences:
            tokens = my_tokenizer(sentence)
            for t in tokens:
                if t not in word2idx:
                    word_freq[t] = 0
                    word2idx[t] = idx
                    idx2word.append(t)
                    idx += 1
                word_freq[t] += 1
                word_count += 1

            # create input -> output context
            for i in xrange(len(tokens)):
                # look at [i-mid..i+mid]
                idx1 = word2idx[tokens[i]]
                X.append(idx1)
                y = []
                for j in xrange(max(0, i - mid), min(i + mid, len(tokens))):
                    if i != j:
                        # Do NOT try to use pairs of strings -- too much RAM
                        idx2 = word2idx[tokens[j]]
                        y.append(idx2)
                Y.append(y)

            num_sentences += 1
            if num_sentences % 100 == 0:
                print "current num sentences:", num_sentences
            # tmp:
            if num_sentences >= 1000:
                break

        N = len(X)
        print "num training pairs:", N
        
        V = idx # vocabulary size
        print "vocabulary size:", V

        # for negative sampling
        self.Pnw = np.zeros(V)
        for i in xrange(V):
            t = idx2word[i]
            self.Pnw[i] = (word_freq[t] / float(word_count))**0.75

        self.word2idx = word2idx
        self.idx2word = idx2word

        # initialize weights
        W1_0 = init_weights((V, self.M)).astype(np.float32)
        W2_0 = init_weights((self.M, V)).astype(np.float32)

        self.fit_np(X, Y, W1_0, W2_0, learning_rate, mu, reg, epochs, batch_sz)
        # self.fit_theano(X, Y, V, learning_rate, mu, reg, epochs, batch_sz)


    # def fit_np(self, X, Y, W1_0, W2_0, learning_rate, mu, reg, epochs, batch_sz):
    #     reg = 0 # temp
    #     N = len(X)
    #     V = W1_0.shape[0]

    #     self.W1 = W1_0
    #     self.W2 = W2_0
    #     dW1 = np.zeros(W1_0.shape)
    #     dW2 = np.zeros(W2_0.shape)

    #     # n_batches = N / batch_sz
    #     costs = []
    #     sample_indices = range(N)
    #     for i in xrange(epochs):
    #         # X, Y = shuffle(X, Y)
    #         sample_indices = shuffle(sample_indices)
    #         for it in xrange(N):
    #             j = sample_indices[it]
    #             # Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
    #             # Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                
    #             # do the updates manually
    #             Z = self.W1[X[j],:] # note: paper uses linear activation function
    #             posA = Z.dot(self.W2[:,Y[j]])
    #             pos_pY = sigmoid(posA)

    #             # temporarily save context values because we don't want to negative sample these
    #             saved = []
    #             for yj in Y[j]:
    #                 saved.append(self.Pnw[yj])
    #                 self.Pnw[yj] = 0
    #             neg_samples = np.random.choice(
    #                 xrange(V),
    #                 size=10, # this is arbitrary
    #                 replace=False,
    #                 p=self.Pnw / np.sum(self.Pnw),
    #             )
    #             for k, yj in zip(xrange(len(saved)), Y[j]):
    #                 self.Pnw[yj] = saved[k]

    #             # technically can remove this line now but leave for sanity checking
    #             neg_samples = np.setdiff1d(neg_samples, Y[j])
    #             # print "number of negative samples:", len(neg_samples)
    #             negA = Z.dot(self.W2[:,neg_samples])
    #             neg_pY = sigmoid(-negA)
    #             c = -np.log(pos_pY).sum() - np.log(neg_pY).sum()

    #             # positive samples
    #             pos_err = pos_pY - 1
    #             # print "pos_err shape:", pos_err.shape, "Z shape:", Z.shape, "dW2[:, Y[j]] shape:", dW2[:, Y[j]].shape
    #             dW2[:, Y[j]] = mu*dW2[:, Y[j]] - learning_rate*(np.outer(Z, pos_err) + reg*self.W2[:, Y[j]])

    #             # negative samples
    #             neg_err = 1 - neg_pY
    #             dW2[:, neg_samples] = mu*dW2[:, neg_samples] - learning_rate*(Z.T.dot(neg_err) + reg*self.W2[:, neg_samples])

    #             self.W2[:, Y[j]] += dW2[:, Y[j]]
    #             self.W2[:, Y[j]] /= np.linalg.norm(self.W2[:, Y[j]], axis=1, keepdims=True)
    #             self.W2[:, neg_samples] += dW2[:, neg_samples]
    #             self.W2[:, neg_samples] /= np.linalg.norm(self.W2[:, neg_samples], axis=1, keepdims=True)

    #             # input weights
    #             gradW1 = pos_err.dot(self.W2[:, Y[j]].T) + neg_err.dot(self.W2[:, neg_samples].T)
    #             dW1[X[j], :] = mu*dW1[X[j], :] - learning_rate*(gradW1 + reg*self.W1[X[j], :])

    #             self.W1[X[j], :] += dW1[X[j], :]
    #             self.W1[X[j], :] /= np.linalg.norm(self.W1[X[j], :])

    #             costs.append(c)
    #             if it % 100 == 0:
    #                 print "cost at j:", j, "/", N, ":", c
    #     plt.plot(costs)
    #     plt.show()

    def fit(self, X, learning_rate=10e-5, mu=0.99, reg=0.1, epochs=10):
        N = len(X)
        V = self.V
        D = self.D

        # calculate Pn(w) - probability distribution for negative sampling
        # basically just the word probability ^ 3/4
        word_freq = {}
        word_count = sum(len(x) for x in X)
        for x in X:
            for xj in x:
                if xj not in word_freq:
                    word_freq[xj] = 0
                word_freq[xj] += 1
        self.Pnw = np.zeros(V)
        for j in xrange(2, V): # 0 and 1 are the start and end tokens, we won't use those here
            self.Pnw[j] = (word_freq[j] / float(word_count))**0.75

        # initialize weights and momentum changes
        self.W1 = init_weights((V, D))
        self.W2 = init_weights((D, V))
        dW1 = np.zeros(self.W1.shape)
        dW2 = np.zeros(self.W2.shape)

        costs = []
        sample_indices = range(N)
        for i in xrange(epochs):
            sample_indices = shuffle(sample_indices)
            for it in xrange(N):
                j = sample_indices[it]
                x = X[j] # one sentence
                cj = 0
                for jj in xrange(self.context_sz, len(x) - self.context_sz):
                
                    # do the updates manually
                    Z = self.W1[x[jj],:] # note: paper uses linear activation function
                    context = np.concatenate([x[(jj - self.context_sz):jj], x[(jj+1):(jj+1+self.context_sz)]])
                    posA = Z.dot(self.W2[:,context])
                    pos_pY = sigmoid(posA)

                    # temporarily save context values because we don't want to negative sample these
                    saved = {}
                    for jjj in xrange(jj - self.context_sz, jj):
                        saved[jjj] = self.Pnw[jjj]
                        self.Pnw[jjj] = 0
                    for jjj in xrange(jj + 1, jj + 1 + self.context_sz):
                        saved[jjj] = self.Pnw[jjj]
                        self.Pnw[jjj] = 0
                    neg_samples = np.random.choice(
                        xrange(V),
                        size=10, # this is arbitrary - number of negative samples to take
                        replace=False,
                        p=self.Pnw / np.sum(self.Pnw),
                    )
                    for jjj, pnwjjj in saved.iteritems():
                        self.Pnw[jjj] = pnwjjj

                    # technically can remove this line now but leave for sanity checking
                    # neg_samples = np.setdiff1d(neg_samples, Y[j])
                    # print "number of negative samples:", len(neg_samples)
                    negA = Z.dot(self.W2[:,neg_samples])
                    neg_pY = sigmoid(-negA)
                    c = -np.log(pos_pY).sum() - np.log(neg_pY).sum()
                    cj += c

                    # positive samples
                    pos_err = pos_pY - 1
                    # print "pos_err shape:", pos_err.shape, "Z shape:", Z.shape, "dW2[:, Y[j]] shape:", dW2[:, Y[j]].shape
                    dW2[:, context] = mu*dW2[:, context] - learning_rate*(np.outer(Z, pos_err) + reg*self.W2[:, context])

                    # negative samples
                    neg_err = 1 - neg_pY
                    dW2[:, neg_samples] = mu*dW2[:, neg_samples] - learning_rate*(Z.T.dot(neg_err) + reg*self.W2[:, neg_samples])

                    self.W2[:, context] += dW2[:, context]
                    # self.W2[:, context] /= np.linalg.norm(self.W2[:, context], axis=1, keepdims=True)
                    self.W2[:, neg_samples] += dW2[:, neg_samples]
                    # self.W2[:, neg_samples] /= np.linalg.norm(self.W2[:, neg_samples], axis=1, keepdims=True)

                    # input weights
                    gradW1 = pos_err.dot(self.W2[:, context].T) + neg_err.dot(self.W2[:, neg_samples].T)
                    dW1[X[j], :] = mu*dW1[x[jj], :] - learning_rate*(gradW1 + reg*self.W1[x[jj], :])

                    self.W1[x[jj], :] += dW1[x[jj], :]
                    # self.W1[x[jj], :] /= np.linalg.norm(self.W1[x[jj], :])

                costs.append(cj)
                if it % 100 == 0:
                    print "epoch:", i, "j:", it, "/", N, "cost:", cj
        plt.plot(costs)
        plt.show()

    # def fit_theano(self, X, Y, W1_0, W2_0, learning_rate, mu, reg, epochs, batch_sz):
    #     N = len(X)
    #     V = W1_0.shape[0]
        
    #     self.W1 = theano.shared(W1_0, 'W1')
    #     self.W2 = theano.shared(W2_0, 'W2')
    #     self.params = [self.W1, self.W2]

    #     # momentum updates
    #     dW1 = theano.shared(np.zeros(W1_0.shape, dtype=np.float32), 'dW1')
    #     dW2 = theano.shared(np.zeros(W2_0.shape, dtype=np.float32), 'dW2')
    #     self.dparams = [dW1, dW2]

    #     # X = np.zeros((N, V), dtype=np.float32)
    #     # Y = np.zeros(N)
    #     # for n in xrange(N):
    #     #     p = pairs[n]
    #     #     i = word2idx[p[0]]
    #     #     X[n,i] = 1
    #     #     j = word2idx[p[1]]
    #     #     Y[n] = j

    #     # Do NOT try to use a full matrix X of one-hot vectors -- too much RAM
    #     thX = T.ivector('inputs')
    #     thY = T.ivector('targets')

    #     # can't do straight dot product since we don't have full matrix X
    #     W1_stack = []
    #     for i in xrange(batch_sz):
    #         W1_stack.append(self.W1[thX[i], :])
    #     W1_stack = T.stack(W1_stack)
    #     Z = W1_stack # linear activation
    #     pY = T.nnet.softmax(Z.dot(self.W2))

    #     # rcost = reg*T.sum([(p**2).sum() for p in self.params])
    #     cost = -T.log(pY[T.arange(pY.shape[0]), thY]).sum() #+ rcost

    #     update_W1 = self.W1 - learning_rate*T.grad(cost, self.W1)
    #     update_W1 = update_W1 / update_W1.norm(2, axis=1).T
    #     update_W2 = self.W2 - learning_rate*T.grad(cost, self.W2)
    #     update_W2 = update_W2 / update_W2.norm(2, axis=1).T
    #     updates = [(self.W1, update_W1), (self.W2, update_W2)]

    #     # updates = [
    #     #     (p, p + mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
    #     # ] + [
    #     #     (dp, mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
    #     # ]

    #     train_op = theano.function(
    #         inputs=[thX, thY],
    #         outputs=cost,
    #         updates=updates,
    #     )

    #     n_batches = N / batch_sz
    #     costs = []
    #     for i in xrange(epochs):
    #         X, Y = shuffle(X, Y)
    #         for j in xrange(n_batches):
    #             Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
    #             Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
    #             c = train_op(Xbatch, Ybatch)
    #             costs.append(c)
    #             if j % 10 == 0:
    #                 print "cost at j:", j, "/", n_batches, ":", c
    #     plt.plot(costs)
    #     plt.show()

    def save(self, fn):
        arrays = [self.W1, self.W2]
        np.savez(fn, *arrays)


def main():
    sentences, word2idx = get_wikipedia_data(n_files=1, n_vocab=2000)
    V = len(word2idx)
    model = Model(10, V, 2)
    # fp = open('/Users/macuser/Code/word2vec-proto/wiki.en.text')
    # model.fit(fp)
    model.fit(sentences)
    model.save('w2v_model.npz')


if __name__ == '__main__':
    main()




