from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from scipy.io import loadmat


def getKaggleMNIST():
    # https://www.kaggle.com/c/digit-recognizer
    return getMNISTFormat('../large_files/train.csv')


def getKaggleFashionMNIST():
    # https://www.kaggle.com/zalando-research/fashionmnist
    return getMNISTFormat('../large_files/fashionmnist/fashion-mnist_train.csv')

def getMNISTFormat(path):
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv(path).values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000,1:] / 255.0
    Ytrain = train[:-1000,0].astype(np.int32)

    Xtest  = train[-1000:,1:] / 255.0
    Ytest  = train[-1000:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest

def getKaggleMNIST3D():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    Xtrain = Xtrain.reshape(-1, 28, 28, 1)
    Xtest = Xtest.reshape(-1, 28, 28, 1)
    return Xtrain, Ytrain, Xtest, Ytest

def getKaggleFashionMNIST3D():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleFashionMNIST()
    Xtrain = Xtrain.reshape(-1, 28, 28, 1)
    Xtest = Xtest.reshape(-1, 28, 28, 1)
    return Xtrain, Ytrain, Xtest, Ytest

def getCIFAR10():
    Xtrain = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
    Ytrain = np.zeros(50000, dtype=np.uint8)

    # train data
    for i in range(5):
        fn = 'data_batch_%s.mat' % (i+1)
        d = loadmat('../large_files/cifar-10-batches-mat/' + fn)
        x = d['data']
        y = d['labels'].flatten()
        x = x.reshape(10000, 3, 32, 32)
        x = np.transpose(x, (0, 2, 3, 1))
        Xtrain[i*10000:(i+1)*10000] = x
        Ytrain[i*10000:(i+1)*10000] = y

    # test data
    d = loadmat('../large_files/cifar-10-batches-mat/test_batch.mat')
    x = d['data']
    y = d['labels'].flatten()
    x = x.reshape(10000, 3, 32, 32)
    x = np.transpose(x, (0, 2, 3, 1))
    Xtest = x
    Ytest = y

    return Xtrain, Ytrain, Xtest, Ytest

