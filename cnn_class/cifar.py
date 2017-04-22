# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
import os
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

# from util import getImageData, error_rate, init_weight_and_bias, init_filter
# from ann_theano import HiddenLayer

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def image2array(im):
    arr = np.array(im) # will be (H, W, 3)
    return arr.transpose((2, 1, 0))


def getImageData():
    N = 50000
    savedXpath = '../large_files/cifar10/train_all.npy'
    if not os.path.exists(savedXpath):
        X = np.zeros((N, 3, 32, 32))
        for i in xrange(N):
            im = Image.open("../large_files/cifar10/train/%s.png" % (i + 1))
            X[i] = image2array(im)
            if i % 1000 == 0:
                print i
        np.save(savedXpath, X.astype(np.uint8))
    else:
        X = np.load(savedXpath)
    X = X.astype(np.float32) / 255.0

    # load labels
    Y = np.zeros(N)
    df = pd.read_csv('../large_files/cifar10/trainLabels.csv')
    S = df['label'].tolist()
    idx = 0
    label2idx = {}
    i = 0
    for s in S:
        if s not in label2idx:
            label2idx[s] = idx
            idx += 1
        Y[i] = label2idx[s]
        i += 1
    print "done loading data"
    X, Y = shuffle(X, Y)
    return X[:30000], Y[:30000]


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)


class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2, 2)):
        # mi = input feature map size
        # mo = output feature map size
        sz = (mo, mi, fw, fh)
        W0 = init_filter(sz, poolsz)
        self.W = theano.shared(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = theano.shared(b0)
        self.poolsz = poolsz
        self.params = [self.W, self.b]

    def forward(self, X):
        conv_out = conv2d(input=X, filters=self.W)
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsz,
            ignore_border=True
        )
        return T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, lr=1e-4, mu=0.99, reg=1e-6, decay=0.99999, eps=1e-2, batch_sz=30, epochs=100, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        # initialize convpool layers
        N, c, width, height = X.shape
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layers.append(layer)
            outw = (outw - fw + 1) / 2
            outh = (outh - fh + 1) / 2
            mi = mo

        # initialize mlp layers
        K = len(set(Y))
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh # size must be same as output of last convpool layer
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # logistic regression layer
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for c in self.convpool_layers:
            self.params += c.params
        for h in self.hidden_layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # for rmsprop
        cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # set up theano functions and variables
        thX = T.tensor4('X', dtype='float32')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)

        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])


        # momentum only
        updates = [
            (p, p + mu*dp - lr*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ] + [
            (dp, mu*dp - lr*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        n_batches = N / batch_sz
        costs = []
        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            for j in xrange(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print "i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z = Z.flatten(ndim=2)
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis=1)


def main():
    X, Y = getImageData()
    model = CNN(
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )
    model.fit(X, Y)

if __name__ == '__main__':
    main()
