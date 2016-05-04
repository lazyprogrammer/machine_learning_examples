
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
# TODO: add batch training
# TODO: add ability to switch between datasets

def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
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
            go_backwards=go_backwards
        )
        return h



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
        x = X[:,i*hp:(i*hp + hp),:].dimshuffle((1, 0, 2)).flatten().reshape((w/wp, X.shape[0]*wp*hp))
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



def main(ReUnit=GRU):
    t0 = datetime.now()
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    train = pd.read_csv('../large_files/train.csv').as_matrix()
    train = shuffle(train)

    Xtrain = rearrange( train[:-1000,1:] )
    Ytrain = train[:-1000,0]
    Ytrain_ind  = y2indicator(Ytrain)

    Xtest  = rearrange( train[-1000:,1:] )
    Ytest  = train[-1000:,0]
    Ytest_ind  = y2indicator(Ytest)


    max_iter = 8
    print_period = 200

    lr = np.float32(0.05)
    reg = np.float32(0.0001)
    mu = np.float32(0.99)

    N = Xtrain.shape[0]
    C = Xtrain.shape[1]
    M = 1000
    K = 10

    # New
    wp, hp = 2, 2

    M1 = 256 # num feature maps
    # Wx1_shape = (M1, Xtrain.shape[1]*wp*hp)
    # Wx1_init = init_filter(Wx1_shape)
    # Wh1_init = init_filter( (M1,M1) )
    # bh1_init = np.zeros((M1,), dtype=np.float32)
    # H01_init = init_filter( (M1,) )
    # Wx2_init = init_filter(Wx1_shape)
    # Wh2_init = init_filter( (M1,M1) )
    # bh2_init = np.zeros((M1,), dtype=np.float32)
    # H02_init = init_filter( (M1,) )
    rnn1 = ReUnit('1', 2, 2, C, M1)
    rnn2 = ReUnit('2', 2, 2, C, M1)

    M2 = 256 # num feature maps
    # Wx3_shape = (M2, 2*M1*1*1)
    # Wx3_init = init_filter(Wx3_shape)
    # Wh3_init = init_filter( (M2,M2) )
    # bh3_init = np.zeros((M2,), dtype=np.float32)
    # H03_init = init_filter( (M2,) )
    # Wx4_init = init_filter(Wx3_shape)
    # Wh4_init = init_filter( (M2,M2) )
    # bh4_init = np.zeros((M2,), dtype=np.float32)
    # H04_init = init_filter( (M2,) )
    rnn3 = ReUnit('3', 1, 1, 2*M1, M2)
    rnn4 = ReUnit('4', 1, 1, 2*M1, M2)

    M3 = 64
    # Wx5_shape = (M3, 2*M2*wp*hp)
    # Wx5_init = init_filter(Wx5_shape)
    # Wh5_init = init_filter( (M3,M3) )
    # bh5_init = np.zeros((M3,), dtype=np.float32)
    # H05_init = init_filter( (M3,) )
    # Wx6_init = init_filter(Wx5_shape)
    # Wh6_init = init_filter( (M3,M3) )
    # bh6_init = np.zeros((M3,), dtype=np.float32)
    # H06_init = init_filter( (M3,) )
    rnn5 = ReUnit('5', 2, 2, 2*M2, M3)
    rnn6 = ReUnit('6', 2, 2, 2*M2, M3)

    M4 = 64
    # Wx7_shape = (M4, 2*M3*1*1)
    # Wx7_init = init_filter(Wx7_shape)
    # Wh7_init = init_filter( (M4,M4) )
    # bh7_init = np.zeros((M4,), dtype=np.float32)
    # H07_init = init_filter( (M4,) )
    # Wx8_init = init_filter(Wx7_shape)
    # Wh8_init = init_filter( (M4,M4) )
    # bh8_init = np.zeros((M4,), dtype=np.float32)
    # H08_init = init_filter( (M4,) )
    rnn7 = ReUnit('7', 1, 1, 2*M3, M4)
    rnn8 = ReUnit('8', 1, 1, 2*M3, M4)


    # vanilla ANN weights
    W9_init = np.random.randn(2*M4*7*7, M) / np.sqrt(2*M4*7*7 + M)
    b9_init = np.zeros(M, dtype=np.float32)
    W10_init = np.random.randn(M, K) / np.sqrt(M + K)
    b10_init = np.zeros(K, dtype=np.float32)


    # step 2: define theano variables and expressions
    X = T.tensor3('X', dtype='float32')
    Y = T.matrix('T')
    # Wx1 = theano.shared(Wx1_init, 'Wx1')
    # Wh1 = theano.shared(Wh1_init, 'Wh1')
    # bh1 = theano.shared(bh1_init, 'bh1')
    # H01 = theano.shared(H01_init, 'H01')
    # Wx2 = theano.shared(Wx2_init, 'Wx2')
    # Wh2 = theano.shared(Wh2_init, 'Wh2')
    # bh2 = theano.shared(bh2_init, 'bh2')
    # H02 = theano.shared(H02_init, 'H02')

    # Wx3 = theano.shared(Wx3_init, 'Wx3')
    # Wh3 = theano.shared(Wh3_init, 'Wh3')
    # bh3 = theano.shared(bh3_init, 'bh3')
    # H03 = theano.shared(H03_init, 'H03')
    # Wx4 = theano.shared(Wx4_init, 'Wx4')
    # Wh4 = theano.shared(Wh4_init, 'Wh4')
    # bh4 = theano.shared(bh4_init, 'bh4')
    # H04 = theano.shared(H04_init, 'H04')

    # Wx5 = theano.shared(Wx5_init, 'Wx5')
    # Wh5 = theano.shared(Wh5_init, 'Wh5')
    # bh5 = theano.shared(bh5_init, 'bh5')
    # H05 = theano.shared(H05_init, 'H05')
    # Wx6 = theano.shared(Wx6_init, 'Wx6')
    # Wh6 = theano.shared(Wh6_init, 'Wh6')
    # bh6 = theano.shared(bh6_init, 'bh6')
    # H06 = theano.shared(H06_init, 'H06')

    # Wx7 = theano.shared(Wx7_init, 'Wx7')
    # Wh7 = theano.shared(Wh7_init, 'Wh7')
    # bh7 = theano.shared(bh7_init, 'bh7')
    # H07 = theano.shared(H07_init, 'H07')
    # Wx8 = theano.shared(Wx8_init, 'Wx8')
    # Wh8 = theano.shared(Wh8_init, 'Wh8')
    # bh8 = theano.shared(bh8_init, 'bh8')
    # H08 = theano.shared(H08_init, 'H08')

    W9 = theano.shared(W9_init.astype(np.float32), 'W9')
    b9 = theano.shared(b9_init, 'b9')
    W10 = theano.shared(W10_init.astype(np.float32), 'W10')
    b10 = theano.shared(b10_init, 'b10')
    params = [
        # Wx1, Wh1, bh1, H01,
        # Wx2, Wh2, bh2, H02,
        # Wx3, Wh3, bh3, H03,
        # Wx4, Wh4, bh4, H04,
        # Wx5, Wh5, bh5, H05,
        # Wx6, Wh6, bh6, H06,
        # Wx7, Wh7, bh7, H07,
        # Wx8, Wh8, bh8, H08,
        W9, b9, W10, b10,
    ]
    for rnn in (rnn1, rnn2, rnn3, rnn4, rnn5, rnn6, rnn7, rnn8):
        params += rnn.params

    # momentum changes
    # dW1 = theano.shared(np.zeros(W1_init.shape, dtype=np.float32), 'dW1')
    # db1 = theano.shared(np.zeros(b1_init.shape, dtype=np.float32), 'db1')
    # dW2 = theano.shared(np.zeros(W2_init.shape, dtype=np.float32), 'dW2')
    # db2 = theano.shared(np.zeros(b2_init.shape, dtype=np.float32), 'db2')
    # dW3 = theano.shared(np.zeros(W3_init.shape, dtype=np.float32), 'dW3')
    # db3 = theano.shared(np.zeros(b3_init.shape, dtype=np.float32), 'db3')
    # dW4 = theano.shared(np.zeros(W4_init.shape, dtype=np.float32), 'dW4')
    # db4 = theano.shared(np.zeros(b4_init.shape, dtype=np.float32), 'db4')

    # forward pass
    # Z1 = renet_layer_lr(X, Wx1, Wh1, bh1, H01, Wx2, Wh2, bh2, H02, 28, 28, wp, hp)
    Z1 = renet_layer_lr(X, rnn1, rnn2, 28, 28, wp, hp)

    ## TMP: just test the first/second layer ##
    # tmp_op = theano.function(
    #     inputs=[X],
    #     outputs=Z1,
    # )
    # print "Xtrain[0].shape:", Xtrain[0].shape
    # out = tmp_op(Xtrain[0])
    # print "Z1.shape:", out.shape
    # exit()

    # Z2 = renet_layer_ud(Z1, Wx3, Wh3, bh3, H03, Wx4, Wh4, bh4, H04, 14, 14, 1, 1)
    Z2 = renet_layer_ud(Z1, rnn3, rnn4, 14, 14, 1, 1)

    # tmp_op2 = theano.function(
    #     inputs=[X],
    #     outputs=Z2,
    # )
    # out = tmp_op2(Xtrain[0])
    # print "Z2.shape:", out.shape
    # exit()


    # Z3 = renet_layer_lr(Z2, Wx5, Wh5, bh5, H05, Wx6, Wh6, bh6, H06, 14, 14, wp, hp)
    Z3 = renet_layer_lr(Z2, rnn5, rnn6, 14, 14, wp, hp)

    # tmp_op3 = theano.function(
    #     inputs=[X],
    #     outputs=Z3,
    # )
    # out = tmp_op3(Xtrain[0])
    # print "Z3.shape:", out.shape
    # exit()

    # Z4 = renet_layer_ud(Z3, Wx7, Wh7, bh7, H07, Wx8, Wh8, bh8, H08, 7, 7, 1, 1)
    Z4 = renet_layer_ud(Z3, rnn7, rnn8, 7, 7, 1, 1)

    Z5 = relu(Z4.flatten().dot(W9) + b9)
    pY = T.nnet.softmax( Z5.dot(W10) + b10)

    # tmp_op4 = theano.function(
    #     inputs=[X],
    #     outputs=Z4,
    # )
    # out = tmp_op4(Xtrain[0])
    # print "Z4.shape:", out.shape
    # exit()

    # define the cost function and prediction
    # params = (W1, b1, W2, b2, W3, b3, W4, b4)
    reg_cost = reg*np.sum((param*param).sum() for param in params)
    cost = -(Y * T.log(pY)).sum() + reg_cost
    prediction = T.argmax(pY, axis=1)

    # step 3: training expressions and functions
    # update_W1 = W1 + mu*dW1 - lr*T.grad(cost, W1)
    # update_b1 = b1 + mu*db1 - lr*T.grad(cost, b1)
    # update_W2 = W2 + mu*dW2 - lr*T.grad(cost, W2)
    # update_b2 = b2 + mu*db2 - lr*T.grad(cost, b2)
    # update_W3 = W3 + mu*dW3 - lr*T.grad(cost, W3)
    # update_b3 = b3 + mu*db3 - lr*T.grad(cost, b3)
    # update_W4 = W4 + mu*dW4 - lr*T.grad(cost, W4)
    # update_b4 = b4 + mu*db4 - lr*T.grad(cost, b4)
    updates = [(param, param - lr*T.grad(cost, param)) for param in params]

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
    )

    # create another function for this because we want it over the whole dataset
    get_prediction = theano.function(
        inputs=[X, Y],
        outputs=[cost, prediction],
    )

    print "Setup elapsed time:", (datetime.now() - t0)
    t0 = datetime.now()
    LL = []
    t1 = t0
    for i in xrange(max_iter):
        print "i:", i
        for j in xrange(N):
            # print "j:", j
            Xbatch = Xtrain[j,:]
            Ybatch = Ytrain_ind[j:j+1,:]

            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val = 0
                prediction_val = np.zeros(len(Ytest))
                for k in xrange(len(Ytest)):
                    c, p = get_prediction(Xtest[k], Ytest_ind[k:k+1,:])
                    cost_val += c
                    prediction_val[k] = p[0]
                    # print "pred:", p[0], type(p[0]), "target:", Ytest[k], type(Ytest[k])
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
