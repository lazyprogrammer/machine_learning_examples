# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
import os
import json
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle
from word2vec import get_wikipedia_data, find_analogies

# Experiments
# previous results did not make sense b/c X was built incorrectly
# redo b/c b and c were not being added correctly as 2-D objects

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

    def fit(self, sentences, cc_matrix=None, learning_rate=10e-5, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False, use_theano=True):
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
            print "number of sentences to process:", N
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print "processed", it, "/", N
                n = len(sentence)
                for i in xrange(n):
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
                    for j in xrange(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j) # this is +ve
                        X[wi,wj] += points
                        X[wj,wi] += points

                    # right side
                    for j in xrange(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i) # this is +ve
                        X[wi,wj] += points
                        X[wj,wi] += points

            # save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print "max in X:", X.max()

        # weighting
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1

        print "max in f(X):", fX.max()

        # target
        logX = np.log(X + 1)

        print "max in log(X):", logX.max()

        print "time to build co-occurrence matrix:", (datetime.now() - t0)

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        if gd and use_theano:
            thW = theano.shared(W)
            thb = theano.shared(b)
            thU = theano.shared(U)
            thc = theano.shared(c)
            thLogX = T.matrix('logX')
            thfX = T.matrix('fX')

            params = [thW, thb, thU, thc]

            thDelta = thW.dot(thU.T) + T.reshape(thb, (V, 1)) + T.reshape(thc, (1, V)) + mu - thLogX
            thCost = ( thfX * thDelta * thDelta ).sum()

            grads = T.grad(thCost, params)

            updates = [(p, p - learning_rate*g) for p, g in zip(params, grads)]

            train_op = theano.function(
                inputs=[thfX, thLogX],
                updates=updates,
            )

        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in xrange(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = ( fX * delta * delta ).sum()
            costs.append(cost)
            print "epoch:", epoch, "cost:", cost

            if gd:
                # gradient descent method

                if use_theano:
                    train_op(fX, logX)
                    W = thW.get_value()
                    b = thb.get_value()
                    U = thU.get_value()
                    c = thc.get_value()

                else:
                    # update W
                    oldW = W.copy()
                    for i in xrange(V):
                        # for j in xrange(V):
                        #     W[i] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])*U[j]
                        W[i] -= learning_rate*(fX[i,:]*delta[i,:]).dot(U)
                    W -= learning_rate*reg*W
                    # print "updated W"

                    # update b
                    for i in xrange(V):
                        # for j in xrange(V):
                        #     b[i] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])
                        b[i] -= learning_rate*fX[i,:].dot(delta[i,:])
                    b -= learning_rate*reg*b
                    # print "updated b"

                    # update U
                    for j in xrange(V):
                        # for i in xrange(V):
                        #     U[j] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])*W[i]
                        U[j] -= learning_rate*(fX[:,j]*delta[:,j]).dot(oldW)
                    U -= learning_rate*reg*U
                    # print "updated U"

                    # update c
                    for j in xrange(V):
                        # for i in xrange(V):
                        #     c[j] -= learning_rate*fX[i,j]*(W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])
                        c[j] -= learning_rate*fX[:,j].dot(delta[:,j])
                    c -= learning_rate*reg*c
                    # print "updated c"

            else:
                # ALS method

                # update W
                # fast way
                # t0 = datetime.now()
                for i in xrange(V):
                    # matrix = reg*np.eye(D) + np.sum((fX[i,j]*np.outer(U[j], U[j]) for j in xrange(V)), axis=0)
                    matrix = reg*np.eye(D) + (fX[i,:]*U.T).dot(U)
                    # assert(np.abs(matrix - matrix2).sum() < 10e-5)
                    vector = (fX[i,:]*(logX[i,:] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)
                # print "fast way took:", (datetime.now() - t0)

                # slow way
                # t0 = datetime.now()
                # for i in xrange(V):
                #     matrix2 = reg*np.eye(D)
                #     vector2 = 0
                #     for j in xrange(V):
                #         matrix2 += fX[i,j]*np.outer(U[j], U[j])
                #         vector2 += fX[i,j]*(logX[i,j] - b[i] - c[j])*U[j]
                # print "slow way took:", (datetime.now() - t0)

                    # assert(np.abs(matrix - matrix2).sum() < 10e-5)
                    # assert(np.abs(vector - vector2).sum() < 10e-5)
                    # W[i] = np.linalg.solve(matrix, vector)
                # print "updated W"

                # update b
                for i in xrange(V):
                    denominator = fX[i,:].sum()
                    # assert(denominator > 0)
                    numerator = fX[i,:].dot(logX[i,:] - W[i].dot(U.T) - c - mu)
                    # for j in xrange(V):
                    #     numerator += fX[i,j]*(logX[i,j] - W[i].dot(U[j]) - c[j])
                    b[i] = numerator / denominator / (1 + reg)
                # print "updated b"

                # update U
                for j in xrange(V):
                    # matrix = reg*np.eye(D) + np.sum((fX[i,j]*np.outer(W[i], W[i]) for i in xrange(V)), axis=0)
                    matrix = reg*np.eye(D) + (fX[:,j]*W.T).dot(W)
                    # assert(np.abs(matrix - matrix2).sum() < 10e-8)
                    vector = (fX[:,j]*(logX[:,j] - b - c[j] - mu)).dot(W)
                    # matrix = reg*np.eye(D)
                    # vector = 0
                    # for i in xrange(V):
                    #     matrix += fX[i,j]*np.outer(W[i], W[i])
                    #     vector += fX[i,j]*(logX[i,j] - b[i] - c[j])*W[i]
                    U[j] = np.linalg.solve(matrix, vector)
                # print "updated U"

                # update c
                for j in xrange(V):
                    denominator = fX[:,j].sum()
                    numerator = fX[:,j].dot(logX[:,j] - W.dot(U[j]) - b  - mu)
                    # for i in xrange(V):
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


def main(we_file, w2i_file, n_files=50):
    cc_matrix = "cc_matrix_%s.npy" % n_files

    # hacky way of checking if we need to re-load the raw data or not
    # remember, only the co-occurrence matrix is needed for training
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = [] # dummy - we won't actually use it
    else:
        sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove(80, V, 10)
    # model.fit(sentences, cc_matrix=cc_matrix, epochs=20) # ALS
    model.fit(
        sentences,
        cc_matrix=cc_matrix,
        learning_rate=3*10e-5,
        reg=0.01,
        epochs=2000,
        gd=True,
        use_theano=True
    ) # gradient descent
    model.save(we_file)


if __name__ == '__main__':
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'
    main(we, w2i)
    for concat in (True, False):
        print "** concat:", concat
        find_analogies('king', 'man', 'woman', concat, we, w2i)
        find_analogies('france', 'paris', 'london', concat, we, w2i)
        find_analogies('france', 'paris', 'rome', concat, we, w2i)
        find_analogies('paris', 'france', 'italy', concat, we, w2i)
        find_analogies('france', 'french', 'english', concat, we, w2i)
        find_analogies('japan', 'japanese', 'chinese', concat, we, w2i)
        find_analogies('japan', 'japanese', 'italian', concat, we, w2i)
        find_analogies('japan', 'japanese', 'australian', concat, we, w2i)
        find_analogies('december', 'november', 'june', concat, we, w2i)
