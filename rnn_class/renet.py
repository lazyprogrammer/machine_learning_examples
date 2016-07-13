# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import os
import gzip
import cPickle
import urllib2
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from sklearn.utils import shuffle

from datetime import datetime

# TODO: add LSTM
# NOTE: batch training is slow to compile, do it better with slicing
# TODO: use only Ytest in cost instead of indicator
# TODO: multiply X with W replicated so that X.dot(W) is not inside the scan loop

def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10), dtype='int32')
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind


def init_filter(shape):
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w.astype(np.float32)


def rearrange(X):
    N = len(X)
    out = np.zeros((N, 1, 28, 28), dtype=np.float32)
    for i in xrange(N):
        out[i, 0, :, :] = X[i].reshape(28, 28)
    return out / 255


class GRU(object):
    def __init__(self, name, wp, hp, C, M):
        Wx_shape = (M, C*wp*hp)

        Wxr_init = init_filter(Wx_shape)
        Whr_init = init_filter((M, M))
        br_init  = np.zeros((M,), dtype=np.float32)

        Wxz_init = init_filter(Wx_shape)
        Whz_init = init_filter((M, M))
        bz_init  = np.zeros((M,), dtype=np.float32)

        Wxh_init = init_filter(Wx_shape)
        Whh_init = init_filter((M, M) )
        bh_init  = np.zeros((M,), dtype=np.float32)

        H0_init = init_filter((M,))
        # ---
        self.Wxr = theano.shared(Wxr_init, 'Wxr_%s' % name)
        self.Whr = theano.shared(Whr_init, 'Whr_%s' % name)
        self.br  = theano.shared(br_init, 'br_%s' % name)

        self.Wxz = theano.shared(Wxz_init, 'Wxz_%s' % name)
        self.Whz = theano.shared(Whz_init, 'Whz_%s' % name)
        self.bz  = theano.shared(bz_init, 'bz_%s' % name)

        self.Wxh = theano.shared(Wxh_init, 'Wxh_%s' % name)
        self.Whh = theano.shared(Whh_init, 'Whh_%s' % name)
        self.bh  = theano.shared(bh_init, 'bh_%s' % name)

        self.H0 = theano.shared(H0_init, 'H0_%s' % name)
        # ---
        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, self.H0]

    def recurrence(self, x_t, h_tm1):
        r_t = T.nnet.sigmoid( T.dot(self.Wxr, x_t) + T.dot(h_tm1, self.Whr) + self.br)
        z_t = T.nnet.sigmoid( T.dot(self.Wxz, x_t) + T.dot(h_tm1, self.Whz) + self.bz)
        hht = relu( T.dot(self.Wxh, x_t) + T.dot( r_t*h_tm1, self.Whh ) + self.bh)
        h_t = (1 - z_t)*h_tm1 + z_t*hht
        return h_t

    def output(self, x, go_backwards=False):
        # input X should be a matrix (2-D)
        # rows index time
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.H0],
            n_steps=x.shape[0],
            go_backwards=go_backwards
        )
        return h


class RNNUnit(object):
    def __init__(self, name, wp, hp, C, M):
        # C = num input feature maps
        # M = num output feature maps
        # print "NAME:", name
        Wx_shape = (M, C*wp*hp)
        Wx_init = init_filter(Wx_shape)
        Wh_init = init_filter( (M, M) )
        bh_init = np.zeros((M,), dtype=np.float32)
        H0_init = init_filter( (M,) )

        self.Wx = theano.shared(Wx_init, 'Wx_%s' % name)
        self.Wh = theano.shared(Wh_init, 'Wh_%s' % name)
        self.bh = theano.shared(bh_init, 'bh_%s' % name)
        self.H0 = theano.shared(H0_init, 'H0_%s' % name)
        self.params = [self.Wx, self.Wh, self.bh, self.H0]

    def recurrence(self, x_t, h_tm1):
        dot = T.dot(self.Wx, x_t)
        h_t = relu(dot + T.dot(h_tm1, self.Wh) + self.bh)
        return h_t

    def output(self, x, go_backwards=False):
        # input X should be a matrix (2-D)
        # rows index time
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.H0],
            n_steps=x.shape[0],
            go_backwards=go_backwards,
            # non_sequences=self.params,
            # strict=True,
        )
        return h


def renet_layer_lr_noscan(X, rnn1, rnn2, w, h, wp, hp):
    list_of_images = []
    for i in xrange(h/hp):
        # x = X[:,i*hp:(i*hp + hp),:].dimshuffle((2, 0, 1)).flatten().reshape((w/wp, X.shape[0]*wp*hp))
        h_tm1 = rnn1.H0
        hr_tm1 = rnn2.H0
        h1 = []
        h2 = []
        for j in xrange(w/wp):
            x = X[:,i*hp:(i*hp + hp),j*wp:(j*wp + wp)].flatten()
            h_t = rnn1.recurrence(x, h_tm1)
            h1.append(h_t)
            h_tm1 = h_t

            jr = w/wp - j - 1
            xr = X[:,i*hp:(i*hp + hp),jr*wp:(jr*wp + wp)].flatten()
            hr_t = rnn2.recurrence(x, hr_tm1)
            h2.append(hr_t)
            hr_tm1 = hr_t
        img = T.concatenate([h1, h2])
        list_of_images.append(img)
    return T.stacklists(list_of_images).dimshuffle((1, 0, 2))


def renet_layer_lr_allscan(X, rnn1, rnn2, w, h, wp, hp):
    # list_of_images = []
    C = X.shape[0]
    X = X.dimshuffle((1, 0, 2)).reshape((h/hp, hp*C*w)) # split the rows for the first scan
    def rnn_pass(x):
        x = x.reshape((hp, C, w)).dimshuffle((2, 1, 0)).reshape((w/wp, C*wp*hp))
        h1 = rnn1.output(x)
        h2 = rnn2.output(x, go_backwards=True)
        img = T.concatenate([h1.T, h2.T])
        # list_of_images.append(img)
        return img

    results, _ = theano.scan(
        fn=rnn_pass,
        sequences=X,
        outputs_info=None,
        n_steps=h/hp,
    )
    return results.dimshuffle((1, 0, 2))
    # return T.stacklists(list_of_images).dimshuffle((1, 0, 2))


def renet_layer_ud_allscan(X, rnn1, rnn2, w, h, wp, hp):
    return renet_layer_lr_allscan(X.dimshuffle((0, 2, 1)), rnn1, rnn2, w, h, wp, hp)


# expect the input image to be K x width x height
# def renet_layer_lr(X, Wx1, Wh1, Bh1, H01, Wx2, Wh2, Bh2, H02, w, h, wp, hp):
def renet_layer_lr(X, rnn1, rnn2, w, h, wp, hp):
    # def recurrence1(x_t, h_tm1):
    #     dot = T.dot(Wx1, x_t)
    #     h_t = relu(dot + T.dot(h_tm1, Wh1) + Bh1)
    #     return h_t

    # def recurrence2(x_t, h_tm1):
    #     dot = T.dot(Wx2, x_t)
    #     h_t = relu(dot + T.dot(h_tm1, Wh2) + Bh2)
    #     return h_t

    list_of_images = []
    # lefts = []
    # rights = []
    for i in xrange(h/hp):
        x = X[:,i*hp:(i*hp + hp),:].dimshuffle((2, 0, 1)).flatten().reshape((w/wp, X.shape[0]*wp*hp))
        # reshape the row into a 2-D matrix to be fed into scan
        # h1, _ = theano.scan(
        #     fn=recurrence1,
        #     sequences=x,
        #     outputs_info=[H01],
        #     n_steps=x.shape[0]
        # )
        # h2, _ = theano.scan(
        #     fn=recurrence2,
        #     sequences=x,
        #     outputs_info=[H02],
        #     n_steps=x.shape[0],
        #     go_backwards=True
        # )
        h1 = rnn1.output(x)
        h2 = rnn2.output(x, go_backwards=True)
        
        # combine the last values of s1 and s2 into an image
        img = T.concatenate([h1.T, h2.T])
        list_of_images.append(img)
        # lefts.append(s1.T)
        # rights.append(s2.T)

    return T.stacklists(list_of_images).dimshuffle((1, 0, 2))


def renet_layer_ud(X, rnn1, rnn2, w, h, wp, hp):
    # def recurrence1(x_t, h_tm1):
    #     dot = T.dot(Wx1, x_t)
    #     h_t = relu(dot + T.dot(h_tm1, Wh1) + Bh1)
    #     return h_t
    # def recurrence2(x_t, h_tm1):
    #     dot = T.dot(Wx2, x_t)
    #     h_t = relu(dot + T.dot(h_tm1, Wh2) + Bh2)
    #     return h_t

    list_of_images = []
    for j in xrange(w/wp):
        # x = X[:,:,j*wp:(j*wp + wp)].dimshuffle((2, 0, 1)).flatten(ndim=2)
        # reshape the row into a 2-D matrix to be fed into scan
        x = X[:,:,j*wp:(j*wp + wp)].dimshuffle((2, 0, 1)).flatten().reshape((h/hp, X.shape[0]*wp*hp))
        # h1, _ = theano.scan(
        #     fn=recurrence1,
        #     sequences=x,
        #     outputs_info=[H01],
        #     n_steps=x.shape[0]
        # )
        # h2, _ = theano.scan(
        #     fn=recurrence2,
        #     sequences=x,
        #     outputs_info=[H02],
        #     n_steps=x.shape[0],
        #     go_backwards=True
        # )
        h1 = rnn1.output(x)
        h2 = rnn2.output(x, go_backwards=True)
        # combine the last values of s1 and s2 into an image
        img = T.concatenate([h1.T, h2.T])
        list_of_images.append(img)

    return T.stacklists(list_of_images).dimshuffle((1, 0, 2))


def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv('../large_files/train.csv').as_matrix()
    train = shuffle(train)

    Xtrain = rearrange( train[:-1000,1:] )
    Ytrain = train[:-1000,0]
    Ytrain_ind  = y2indicator(Ytrain)

    Xtest  = rearrange( train[-1000:,1:] )
    Ytest  = train[-1000:,0]
    Ytest_ind  = y2indicator(Ytest)
    return Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind


def getMNIST():
    # data shape: train (50000, 784), test (10000, 784)
    # already scaled from 0..1 and converted to float32
    datadir = '../large_files/'
    if not os.path.exists(datadir):
        datadir = ''

    input_file = "%smnist.pkl.gz" % datadir
    if not os.path.exists(input_file):
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        with open(input_file, "wb") as out:
            f = urllib2.urlopen(url)
            out.write(f.read())
            out.flush()

    with gzip.open(input_file) as f:
        train, valid, test = cPickle.load(f)

    Xtrain, Ytrain = train
    Xvalid, Yvalid = valid
    Xtest, Ytest = test

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Xtest, Ytest = shuffle(Xtest, Ytest)

    # try to take a smaller sample
    Xtrain = Xtrain[0:30000]
    Ytrain = Ytrain[0:30000]
    Xtest = Xtest[0:1000]
    Ytest = Ytest[0:1000]

    return Xtrain.reshape(len(Xtrain), 1, 28, 28), Ytrain, Ytrain_ind, Xtest.reshape(len(Xtest), 1, 28, 28), Ytest, Ytest_ind


def main(ReUnit=RNNUnit, getData=getMNIST):
    t0 = datetime.now()
    print "Start time:", t0
    
    Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind = getData()

    max_iter = 8
    print_period = 200

    lr = np.float32(0.05)
    reg = np.float32(0.0001)
    mu = np.float32(0.99)

    N = Xtrain.shape[0]
    C = Xtrain.shape[1]
    M = 300
    K = 10

    batch_sz = 1
    n_batches = N / batch_sz

    M1 = 256 # num feature maps
    rnn1 = ReUnit('1', 2, 2, C, M1)
    rnn2 = ReUnit('2', 2, 2, C, M1)

    M2 = 256 # num feature maps
    rnn3 = ReUnit('3', 1, 1, 2*M1, M2)
    rnn4 = ReUnit('4', 1, 1, 2*M1, M2)

    M3 = 64
    rnn5 = ReUnit('5', 2, 2, 2*M2, M3)
    rnn6 = ReUnit('6', 2, 2, 2*M2, M3)

    M4 = 64
    rnn7 = ReUnit('7', 1, 1, 2*M3, M4)
    rnn8 = ReUnit('8', 1, 1, 2*M3, M4)

    print "Finished creating rnn objects, elapsed time:", (datetime.now() - t0)


    # vanilla ANN weights
    W9_init = np.random.randn(2*M4*7*7, M) / np.sqrt(2*M4*7*7 + M)
    b9_init = np.zeros(M, dtype=np.float32)
    W10_init = np.random.randn(M, K) / np.sqrt(M + K)
    b10_init = np.zeros(K, dtype=np.float32)


    # step 2: define theano variables and expressions
    X = T.tensor4('X', dtype='float32')
    x = T.tensor3('x', dtype='float32')
    Y = T.matrix('T')

    W9 = theano.shared(W9_init.astype(np.float32), 'W9')
    b9 = theano.shared(b9_init, 'b9')
    W10 = theano.shared(W10_init.astype(np.float32), 'W10')
    b10 = theano.shared(b10_init, 'b10')
    params = [W9, b9, W10, b10]
    for rnn in (rnn1, rnn2, rnn3, rnn4, rnn5, rnn6, rnn7, rnn8):
        params += rnn.params


    print "Finished creating all shared vars, elapsed time:", (datetime.now() - t0)
    # momentum changes
    # dW1 = theano.shared(np.zeros(W1_init.shape, dtype=np.float32), 'dW1')
    # db1 = theano.shared(np.zeros(b1_init.shape, dtype=np.float32), 'db1')
    # dW2 = theano.shared(np.zeros(W2_init.shape, dtype=np.float32), 'dW2')
    # db2 = theano.shared(np.zeros(b2_init.shape, dtype=np.float32), 'db2')
    # dW3 = theano.shared(np.zeros(W3_init.shape, dtype=np.float32), 'dW3')
    # db3 = theano.shared(np.zeros(b3_init.shape, dtype=np.float32), 'db3')
    # dW4 = theano.shared(np.zeros(W4_init.shape, dtype=np.float32), 'dW4')
    # db4 = theano.shared(np.zeros(b4_init.shape, dtype=np.float32), 'db4')

    # Z_tmp = renet_layer_lr_allscan(x, rnn1, rnn2, 28, 28, 2, 2)
    # # Z_tmp = renet_layer_lr_noscan(x, rnn1, rnn2, 28, 28, 2, 2)
    # tmp_op = theano.function(
    #     inputs=[x],
    #     outputs=Z_tmp,
    # )
    # print "Xtrain[0].shape:", Xtrain[0].shape
    # out = tmp_op(Xtrain[0])
    # print "Z_tmp.shape:", out.shape
    # exit()

    def forward(x):
        # x = args[0]
        # forward pass
        Z1 = renet_layer_lr_allscan(x, rnn1, rnn2, 28, 28, 2, 2)
        Z2 = renet_layer_ud_allscan(Z1, rnn3, rnn4, 14, 14, 1, 1)
        Z3 = renet_layer_lr_allscan(Z2, rnn5, rnn6, 14, 14, 2, 2)
        Z4 = renet_layer_ud_allscan(Z3, rnn7, rnn8, 7, 7, 1, 1)
        Z5 = relu(Z4.flatten().dot(W9) + b9)
        pY = T.nnet.softmax( Z5.dot(W10) + b10)
        return pY

    if True: #batch_sz > 1:
        batch_forward_out3, _ = theano.scan(
            fn=forward,
            sequences=X,
            # outputs_info=[self.H0],
            n_steps=X.shape[0],
            # non_sequences=params,
            # strict=True,
        )
    else:
        batch_forward_out3 = forward(X[0])

    print "Finished creating output scan, elapsed time:", (datetime.now() - t0)
    batch_forward_out = batch_forward_out3.flatten(ndim=2) # the output will be (N, 1, 10)

    print "Finished reshaping output, elapsed time:", (datetime.now() - t0)

    ## TMP: just test the first/second layer ##
    # tmp_op = theano.function(
    #     inputs=[X],
    #     outputs=Z1,
    # )
    # print "Xtrain[0].shape:", Xtrain[0].shape
    # out = tmp_op(Xtrain[0])
    # print "Z1.shape:", out.shape
    # exit()

    

    # tmp_op2 = theano.function(
    #     inputs=[X],
    #     outputs=Z2,
    # )
    # out = tmp_op2(Xtrain[0])
    # print "Z2.shape:", out.shape
    # exit()

    

    # tmp_op3 = theano.function(
    #     inputs=[X],
    #     outputs=Z3,
    # )
    # out = tmp_op3(Xtrain[0])
    # print "Z3.shape:", out.shape
    # exit()

    

    # tmp_op4 = theano.function(
    #     inputs=[X],
    #     outputs=Z4,
    # )
    # out = tmp_op4(Xtrain[0])
    # print "Z4.shape:", out.shape
    # exit()

    # tmp_op_out = theano.function(inputs=[X], outputs=batch_forward_out)
    # out = tmp_op_out(Xtest[0:50,])
    # print "out.shape:", out.shape
    # exit()

    # define the cost function and prediction
    # params = (W1, b1, W2, b2, W3, b3, W4, b4)
    reg_cost = reg*np.sum((param*param).sum() for param in params)
    cost = -(Y * T.log(batch_forward_out)).sum() + reg_cost
    prediction = T.argmax(batch_forward_out, axis=1)

    # step 3: training expressions and functions
    updates = [(param, param - lr*T.grad(cost, param)) for param in params]

    print "Finished creating update expressions, elapsed time:", (datetime.now() - t0)

    # update weight changes
    # update_dW1 = mu*dW1 - lr*T.grad(cost, W1)
    # update_db1 = mu*db1 - lr*T.grad(cost, b1)
    # update_dW2 = mu*dW2 - lr*T.grad(cost, W2)
    # update_db2 = mu*db2 - lr*T.grad(cost, b2)
    # update_dW3 = mu*dW3 - lr*T.grad(cost, W3)
    # update_db3 = mu*db3 - lr*T.grad(cost, b3)
    # update_dW4 = mu*dW4 - lr*T.grad(cost, W4)
    # update_db4 = mu*db4 - lr*T.grad(cost, b4)

    train = theano.function(
        inputs=[X, Y],
        updates=updates,
        allow_input_downcast=True,
    )

    # create another function for this because we want it over the whole dataset
    get_prediction = theano.function(
        inputs=[X, Y],
        outputs=[cost, prediction],
        allow_input_downcast=True,
    )

    print "Setup elapsed time:", (datetime.now() - t0)

    # test it
    # print get_prediction(Xtest, Ytest_ind)
    # exit()

    t0 = datetime.now()
    LL = []
    t1 = t0
    for i in xrange(max_iter):
        print "i:", i
        for j in xrange(n_batches):
            # print "j:", j
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),:]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),:]

            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                # cost_val = 0
                # prediction_val = np.zeros(len(Ytest))
                # for k in xrange(len(Ytest)):
                #     c, p = get_prediction(Xtest[k], Ytest_ind[k:k+1,:])
                #     cost_val += c
                #     prediction_val[k] = p[0]
                #     # print "pred:", p[0], type(p[0]), "target:", Ytest[k], type(Ytest[k])
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.2f" % (i, j, cost_val / len(Ytest), err)
                t2 = datetime.now()
                print "Time since last print:", (t2 - t1)
                t1 = t2
                LL.append(cost_val)
    print "Elapsed time:", (datetime.now() - t0)
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()
