# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import os
import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle
from util import find_analogies


import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

# using ALS, what's the least # files to get correct analogies?
# use this for word2vec training to make it faster
# first tried 20 files --> not enough
# how about 30 files --> some correct but still not enough
# 40 files --> half right but 50 is better

class Glove:
    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False, use_theano=False, use_tensorflow=False):
        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling
        # the training data X
        # TODO: would it be better to use a sparse matrix?
        t0 = datetime.now()
        V = self.V
        D = self.D

        if not os.path.exists(cc_matrix):
            X = np.zeros((V, V))
            N = len(sentences)
            print("number of sentences to process:", N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("processed", it, "/", N)
                n = len(sentence)
                for i in range(n):
                    # i is not the word index!!!
                    # j is not the word index!!!
                    # i just points to which element of the sequence (sentence) we're looking at
                    wi = sentence[i]

                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    # we can either choose only one side as context, or both
                    # here we are doing both

                    # make sure "start" and "end" tokens are part of some context
                    # otherwise their f(X) will be 0 (denominator in bias update)
                    if i - self.context_sz < 0:
                        points = 1.0 / (i + 1)
                        X[wi,0] += points
                        X[0,wi] += points
                    if i + self.context_sz > n:
                        points = 1.0 / (n - i)
                        X[wi,1] += points
                        X[1,wi] += points

                    # left side
                    for j in range(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j) # this is +ve
                        X[wi,wj] += points
                        X[wj,wi] += points

                    # right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i) # this is +ve
                        X[wi,wj] += points
                        X[wj,wi] += points

            # save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print("max in X:", X.max())

        # weighting
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1

        print("max in f(X):", fX.max())

        # target
        logX = np.log(X + 1)

        print("max in log(X):", logX.max())

        print("time to build co-occurrence matrix:", (datetime.now() - t0))

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()


        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = ( fX * delta * delta ).sum()
            costs.append(cost)
            print("epoch:", epoch, "cost:", cost)

            if gd:
                # gradient descent method
                # update W
                # oldW = W.copy()
                for i in range(V):
                    # for j in range(V):
                    #     W[i] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])*U[j]
                    W[i] -= learning_rate*(fX[i,:]*delta[i,:]).dot(U)
                W -= learning_rate*reg*W
                # print "updated W"

                # update b
                for i in range(V):
                    # for j in range(V):
                    #     b[i] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])
                    b[i] -= learning_rate*fX[i,:].dot(delta[i,:])
                # b -= learning_rate*reg*b
                # print "updated b"

                # update U
                for j in range(V):
                    # for i in range(V):
                    #     U[j] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])*W[i]
                    U[j] -= learning_rate*(fX[:,j]*delta[:,j]).dot(W)
                U -= learning_rate*reg*U
                # print "updated U"

                # update c
                for j in range(V):
                    # for i in range(V):
                    #     c[j] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])
                    c[j] -= learning_rate*fX[:,j].dot(delta[:,j])
                # c -= learning_rate*reg*c
                # print "updated c"

            else:
                # ALS method

                # update W
                # fast way
                # t0 = datetime.now()
                for i in range(V):
                    # matrix = reg*np.eye(D) + np.sum((fX[i,j]*np.outer(U[j], U[j]) for j in range(V)), axis=0)
                    matrix = reg*np.eye(D) + (fX[i,:]*U.T).dot(U)
                    # assert(np.abs(matrix - matrix2).sum() < 1e-5)
                    vector = (fX[i,:]*(logX[i,:] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)
                # print "fast way took:", (datetime.now() - t0)

                # slow way
                # t0 = datetime.now()
                # for i in range(V):
                #     matrix2 = reg*np.eye(D)
                #     vector2 = 0
                #     for j in range(V):
                #         matrix2 += fX[i,j]*np.outer(U[j], U[j])
                #         vector2 += fX[i,j]*(logX[i,j] - b[i] - c[j])*U[j]
                # print "slow way took:", (datetime.now() - t0)

                    # assert(np.abs(matrix - matrix2).sum() < 1e-5)
                    # assert(np.abs(vector - vector2).sum() < 1e-5)
                    # W[i] = np.linalg.solve(matrix, vector)
                # print "updated W"

                # update b
                for i in range(V):
                    denominator = fX[i,:].sum()
                    # assert(denominator > 0)
                    numerator = fX[i,:].dot(logX[i,:] - W[i].dot(U.T) - c - mu)
                    # for j in range(V):
                    #     numerator += fX[i,j]*(logX[i,j] - W[i].dot(U[j]) - c[j])
                    b[i] = numerator / denominator / (1 + reg)
                # print "updated b"

                # update U
                for j in range(V):
                    # matrix = reg*np.eye(D) + np.sum((fX[i,j]*np.outer(W[i], W[i]) for i in range(V)), axis=0)
                    matrix = reg*np.eye(D) + (fX[:,j]*W.T).dot(W)
                    # assert(np.abs(matrix - matrix2).sum() < 1e-8)
                    vector = (fX[:,j]*(logX[:,j] - b - c[j] - mu)).dot(W)
                    # matrix = reg*np.eye(D)
                    # vector = 0
                    # for i in range(V):
                    #     matrix += fX[i,j]*np.outer(W[i], W[i])
                    #     vector += fX[i,j]*(logX[i,j] - b[i] - c[j])*W[i]
                    U[j] = np.linalg.solve(matrix, vector)
                # print "updated U"

                # update c
                for j in range(V):
                    denominator = fX[:,j].sum()
                    numerator = fX[:,j].dot(logX[:,j] - W.dot(U[j]) - b  - mu)
                    # for i in range(V):
                    #     numerator += fX[i,j]*(logX[i,j] - W[i].dot(U[j]) - b[i])
                    c[j] = numerator / denominator / (1 + reg)
                # print "updated c"

        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        # function word_analogies expects a (V,D) matrx and a (D,V) matrix
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)


def main(we_file, w2i_file, use_brown=True, n_files=100):
    if use_brown:
        cc_matrix = "cc_matrix_brown.npy"
    else:
        cc_matrix = "cc_matrix_%s.npy" % n_files

    # hacky way of checking if we need to re-load the raw data or not
    # remember, only the co-occurrence matrix is needed for training
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = [] # dummy - we won't actually use it
    else:
        if use_brown:
            keep_words = set([
                'king', 'man', 'woman',
                'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
                'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
                'australia', 'australian', 'december', 'november', 'june',
                'january', 'february', 'march', 'april', 'may', 'july', 'august',
                'september', 'october',
            ])
            sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=5000, keep_words=keep_words)
        else:
            sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)
        
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove(200, V, 10)

    # alternating least squares method
    model.fit(sentences, cc_matrix=cc_matrix, epochs=20)

    # gradient descent method
    # model.fit(
    #     sentences,
    #     cc_matrix=cc_matrix,
    #     learning_rate=5e-4,
    #     reg=0.1,
    #     epochs=500,
    #     gd=True,
    # )
    model.save(we_file)


if __name__ == '__main__':
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'
    # we = 'glove_model_brown.npz'
    # w2i = 'glove_word2idx_brown.json'
    main(we, w2i, use_brown=False)
    
    # load back embeddings
    npz = np.load(we)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i:w for w,i in word2idx.items()}

    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2


        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)

