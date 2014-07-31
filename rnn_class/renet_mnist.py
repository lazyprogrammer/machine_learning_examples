# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from sklearn.utils import shuffle

from datetime import datetime


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


# expect the input image to be K x width x height
def renet_layer_lr(X, Wx, Wh, Wo, Bh, Bo, H0, w, h, wp, hp):
    def recurrence(x_t, h_tm1):
        dot = T.dot(Wx, x_t)
        h_t = T.tanh(dot + T.dot(h_tm1, Wh) + Bh)
        s_t = T.tanh(T.dot(h_t, Wo) + Bo)
        return [h_t, s_t]

    list_of_images = []
    # lefts = []
    # rights = []
    for i in xrange(h/hp):
        x = X[:,i*hp:(i*hp + hp),:].dimshuffle((1, 0, 2)).flatten().reshape((w/wp, X.shape[0]*wp*hp))
        # reshape the row into a 2-D matrix to be fed into scan
        [h1, s1], _ = theano.scan(
            fn=recurrence,
            sequences=x,
            outputs_info=[H0, None],
            n_steps=x.shape[0]
        )
        [h2, s2], _ = theano.scan(
            fn=recurrence,
            sequences=x,
            outputs_info=[H0, None],
            n_steps=x.shape[0],
            go_backwards=True
        )
        # combine the last values of s1 and s2 into an image
        img = T.concatenate([s1.T, s2.T])
        list_of_images.append(img)
        # lefts.append(s1.T)
        # rights.append(s2.T)

    return T.stacklists(list_of_images).dimshuffle((1, 0, 2))


def renet_layer_ud(X, Wx, Wh, Wo, Bh, Bo, H0, w, h, wp, hp):
    def recurrence(x_t, h_tm1):
        dot = T.dot(Wx, x_t)
        h_t = T.tanh(dot + T.dot(h_tm1, Wh) + Bh)
        s_t = T.tanh(T.dot(h_t, Wo) + Bo)
        return [h_t, s_t]

    list_of_images = []
    for j in xrange(w/wp):
        # x = X[:,:,j*wp:(j*wp + wp)].dimshuffle((2, 0, 1)).flatten(ndim=2)
        # reshape the row into a 2-D matrix to be fed into scan
        x = X[:,:,j*wp:(j*wp + wp)].dimshuffle((2, 0, 1)).flatten().reshape((h/hp, X.shape[0]*wp*hp))
        [h1, s1], _ = theano.scan(
            fn=recurrence,
            sequences=x,
            outputs_info=[H0, None],
            n_steps=x.shape[0]
        )
        [h2, s2], _ = theano.scan(
            fn=recurrence,
            sequences=x,
            outputs_info=[H0, None],
            n_steps=x.shape[0],
            go_backwards=True
        )
        # combine the last values of s1 and s2 into an image
        img = T.concatenate([s1.T, s2.T])
        list_of_images.append(img)

    return T.stacklists(list_of_images).dimshuffle((1, 0, 2))



def main():
    t0 = datetime.now()
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    train = pd.read_csv('../large_files/train.csv').as_matrix()
    train = shuffle(train)

    Xtrain = rearrange( train[:-100,1:] )
    Ytrain = train[:-100,0]
    Ytrain_ind  = y2indicator(Ytrain)

    Xtest  = rearrange( train[-100:,1:] )
    Ytest  = train[-100:,0]
    Ytest_ind  = y2indicator(Ytest)


    max_iter = 8
    print_period = 200

    lr = np.float32(0.01)
    reg = np.float32(0.01)
    mu = np.float32(0.99)

    N = Xtrain.shape[0]

    M = 4096
    K = 10

    # New
    wp, hp = 2, 2

    M1 = 64 # hidden layer size
    M2 = 256 # num feature maps
    W1x_shape = (M1, 1*wp*hp)
    W1x_init = init_filter(W1x_shape)
    W1h_init = init_filter( (M1,M1) )
    b1h_init = np.zeros((M1,), dtype=np.float32)
    W1o_init = init_filter( (M1, M2) )
    b1o_init = np.zeros((M2,), dtype=np.float32)
    H01_init = init_filter( (M1,) )

    M3 = 64 # hidden layer size
    M4 = 256 # num feature maps
    W2x_shape = (M3, 2*M2*1*1) # TODO: revert
    W2x_init = init_filter(W2x_shape)
    W2h_init = init_filter( (M3,M3) )
    b2h_init = np.zeros((M3,), dtype=np.float32)
    W2o_init = init_filter( (M3, M4) )
    b2o_init = np.zeros((M4,), dtype=np.float32)
    H02_init = init_filter( (M3,) )

    M5 = 64
    M6 = 256
    W3x_shape = (M5, 2*M4*wp*hp)
    W3x_init = init_filter(W3x_shape)
    W3h_init = init_filter( (M5,M5) )
    b3h_init = np.zeros((M5,), dtype=np.float32)
    W3o_init = init_filter( (M5, M6) )
    b3o_init = np.zeros((M6,), dtype=np.float32)
    H03_init = init_filter( (M5,) )

    M7 = 64
    M8 = 256
    W4x_shape = (M7, 2*M6*1*1)
    W4x_init = init_filter(W4x_shape)
    W4h_init = init_filter( (M7,M7) )
    b4h_init = np.zeros((M7,), dtype=np.float32)
    W4o_init = init_filter( (M7, M8) )
    b4o_init = np.zeros((M8,), dtype=np.float32)
    H04_init = init_filter( (M7,) )


    # vanilla ANN weights
    W5_init = np.random.randn(2*M8*7*7, M) / np.sqrt(2*M8*7*7 + M)
    b5_init = np.zeros(M, dtype=np.float32)
    W6_init = np.random.randn(M, K) / np.sqrt(M + K)
    b6_init = np.zeros(K, dtype=np.float32)


    # step 2: define theano variables and expressions
    X = T.tensor3('X', dtype='float32')
    Y = T.matrix('T')
    W1x = theano.shared(W1x_init, 'W1x')
    W1h = theano.shared(W1h_init, 'W1h')
    b1h = theano.shared(b1h_init, 'b1h')
    W1o = theano.shared(W1o_init, 'W1o')
    b1o = theano.shared(b1o_init, 'b1o')
    H01 = theano.shared(H01_init, 'H01')
    W2x = theano.shared(W2x_init, 'W2x')
    W2h = theano.shared(W2h_init, 'W2h')
    b2h = theano.shared(b2h_init, 'b2h')
    W2o = theano.shared(W2o_init, 'W2o')
    b2o = theano.shared(b2o_init, 'b2o')
    H02 = theano.shared(H02_init, 'H02')
    W3x = theano.shared(W3x_init, 'W3x')
    W3h = theano.shared(W3h_init, 'W3h')
    b3h = theano.shared(b3h_init, 'b3h')
    W3o = theano.shared(W3o_init, 'W3o')
    b3o = theano.shared(b3o_init, 'b3o')
    H03 = theano.shared(H03_init, 'H03')
    W4x = theano.shared(W4x_init, 'W4x')
    W4h = theano.shared(W4h_init, 'W4h')
    b4h = theano.shared(b4h_init, 'b4h')
    W4o = theano.shared(W4o_init, 'W4o')
    b4o = theano.shared(b4o_init, 'b4o')
    H04 = theano.shared(H04_init, 'H04')
    W5 = theano.shared(W5_init.astype(np.float32), 'W5')
    b5 = theano.shared(b5_init, 'b5')
    W6 = theano.shared(W6_init.astype(np.float32), 'W6')
    b6 = theano.shared(b6_init, 'b6')
    params = [W1x, W1h, b1h, W1o, b1o, H01, W2x, W2h, b2h, W2o, b2o, H02, W3x, W3h, b3h, W3o, b3o, H03, W4x, W4h, b4h, W4o, b4o, H04, W5, b5, W6, b6]

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
    Z1 = renet_layer_lr(X, W1x, W1h, W1o, b1h, b1o, H01, 28, 28, wp, hp)

    ## TMP: just test the first/second layer ##
    # tmp_op = theano.function(
    #     inputs=[X],
    #     outputs=Z1,
    # )
    # print "Xtrain[0].shape:", Xtrain[0].shape
    # out = tmp_op(Xtrain[0])
    # print "Z1.shape:", out.shape

    Z2 = renet_layer_ud(Z1, W2x, W2h, W2o, b2h, b2o, H02, 14, 14, 1, 1)

    # tmp_op2 = theano.function(
    #     inputs=[X],
    #     outputs=Z2,
    # )
    # out = tmp_op2(Xtrain[0])
    # print "Z2.shape:", out.shape
    # exit()


    Z3 = renet_layer_lr(Z2, W3x, W3h, W3o, b3h, b3o, H03, 14, 14, wp, hp)
    Z4 = renet_layer_ud(Z3, W4x, W4h, W4o, b4h, b4o, H04, 7, 7, 1, 1)
    Z5 = relu(Z4.flatten().dot(W5) + b5)
    pY = T.nnet.softmax( Z5.dot(W6) + b6)


    # tmp_op3 = theano.function(
    #     inputs=[X],
    #     outputs=Z3,
    # )
    # out = tmp_op3(Xtrain[0])
    # print "Z3.shape:", out.shape

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
                prediction_val = np.zeros(100)
                for k in xrange(100):
                    c, p = get_prediction(Xtest[k], Ytest_ind[k:k+1,:])
                    cost_val += c
                    prediction_val[k] = p
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
