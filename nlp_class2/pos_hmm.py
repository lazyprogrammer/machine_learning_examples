# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath('..'))
from hmm_class.hmmd_scaled import HMM

#from pos_baseline import get_data
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score


def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct)/n_total


def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()


# def flatten(l):
#     return [item for sublist in l for item in sublist]

def get_data(split_sequences=False):
    if not os.path.exists('chunking'):
        print("Please create a folder in your local directory called 'chunking'")
        print("train.txt and test.txt should be stored in there.")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('chunking/train.txt'):
        print("train.txt is not in chunking/train.txt")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('chunking/test.txt'):
        print("test.txt is not in chunking/test.txt")
        print("Please check the comments to get the download link.")
        exit()

    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open('chunking/train.txt', encoding='utf-8'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])
            
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    # load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []
    for line in open('chunking/test.txt', encoding='utf-8'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx) # use this as unknown
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx

def main(smoothing=1e-1):
    # X = words, Y = POS tags
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
    V = len(word2idx) + 1

    # find hidden state transition matrix and pi
    M = max(max(y) for y in Ytrain) + 1 #len(set(flatten(Ytrain)))
    A = np.ones((M, M))*smoothing # add-one smoothing
    pi = np.zeros(M)
    for y in Ytrain:
        pi[y[0]] += 1
        for i in range(len(y)-1):
            A[y[i], y[i+1]] += 1
    # turn it into a probability matrix
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    # find the observation matrix
    B = np.ones((M, V))*smoothing # add-one smoothing
    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1
    B /= B.sum(axis=1, keepdims=True)

    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # get predictions
    Ptrain = []
    for x in Xtrain:
        p = hmm.get_state_sequence(x)
        Ptrain.append(p)

    Ptest = []
    for x in Xtest:
        p = hmm.get_state_sequence(x)
        Ptest.append(p)

    # print results
    print("train accuracy:", accuracy(Ytrain, Ptrain))
    print("test accuracy:", accuracy(Ytest, Ptest))
    print("train f1:", total_f1_score(Ytrain, Ptrain))
    print("test f1:", total_f1_score(Ytest, Ptest))

if __name__ == '__main__':
    main()
