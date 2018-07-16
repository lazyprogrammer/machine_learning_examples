# New concepts and differences from Theano:
# - stride is the interval at which to apply the convolution
# - unlike previous course, we use constant-size input to the network
#   since not doing that caused us to start swapping
# - the output after convpool is a different size (8,8) here, (5,5) in Theano

# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.signal import convolve2d
from scipy.io import loadmat
from sklearn.utils import shuffle

from benchmark import get_data, error_rate


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape, poolsz):
    # w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2]) / np.prod(poolsz))
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)


def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3)
    # N = X.shape[-1]
    # out = np.zeros((N, 32, 32, 3), dtype=np.float32)
    # for i in xrange(N):
    #     for j in xrange(3):
    #         out[i, :, :, j] = X[:, :, j, i]
    # return out / 255
    return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)


def main():
    train, test = get_data()

    # Need to scale! don't leave as 0..255
    # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calculation
    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    # print len(Ytrain)
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    Xtest  = rearrange(test['X'])
    Ytest  = test['y'].flatten() - 1
    del test

    # gradient descent params
    max_iter = 6
    print_period = 10
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz

    # limit samples since input will always have to be same size
    # you could also just do N = N / batch_sz * batch_sz
    Xtrain = Xtrain[:73000,]
    Ytrain = Ytrain[:73000]
    Xtest = Xtest[:26000,]
    Ytest = Ytest[:26000]
    # print "Xtest.shape:", Xtest.shape
    # print "Ytest.shape:", Ytest.shape

    # initial weights
    M = 500
    K = 10
    poolsz = (2, 2)

    W1_shape = (5, 5, 3, 20) # (filter_width, filter_height, num_color_channels, num_feature_maps)
    W1_init = init_filter(W1_shape, poolsz)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32) # one bias per output feature map

    W2_shape = (5, 5, 20, 50) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # vanilla ANN weights
    W3_init = np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt(W2_shape[-1]*8*8 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)


    # define variables and expressions
    # using None as the first shape element takes up too much RAM unfortunately
    X = tf.placeholder(tf.float32, shape=(batch_sz, 32, 32, 3), name='X')
    T = tf.placeholder(tf.int32, shape=(batch_sz,), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3 )
    Yish = tf.matmul(Z3, W4) + b4

    cost = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=Yish,
            labels=T
        )
    )

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    t0 = datetime.now()
    LL = []
    W1_val = None
    W2_val = None
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),]

                if len(Xbatch) == batch_sz:
                    session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                    if j % print_period == 0:
                        # due to RAM limitations we need to have a fixed size input
                        # so as a result, we have this ugly total cost and prediction computation
                        test_cost = 0
                        prediction = np.zeros(len(Xtest))
                        for k in range(len(Xtest) // batch_sz):
                            Xtestbatch = Xtest[k*batch_sz:(k*batch_sz + batch_sz),]
                            Ytestbatch = Ytest[k*batch_sz:(k*batch_sz + batch_sz),]
                            test_cost += session.run(cost, feed_dict={X: Xtestbatch, T: Ytestbatch})
                            prediction[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(
                                predict_op, feed_dict={X: Xtestbatch})
                        err = error_rate(prediction, Ytest)
                        print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                        LL.append(test_cost)

        W1_val = W1.eval()
        W2_val = W2.eval()
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()


    W1_val = W1_val.transpose(3, 2, 0, 1)
    W2_val = W2_val.transpose(3, 2, 0, 1)


    # visualize W1 (20, 3, 5, 5)
    # W1_val = W1.get_value()
    grid = np.zeros((8*5, 8*5))
    m = 0
    n = 0
    for i in range(20):
        for j in range(3):
            filt = W1_val[i,j]
            grid[m*5:(m+1)*5,n*5:(n+1)*5] = filt
            m += 1
            if m >= 8:
                m = 0
                n += 1
    plt.imshow(grid, cmap='gray')
    plt.title("W1")
    plt.show()

    # visualize W2 (50, 20, 5, 5)
    # W2_val = W2.get_value()
    grid = np.zeros((32*5, 32*5))
    m = 0
    n = 0
    for i in range(50):
        for j in range(20):
            filt = W2_val[i,j]
            grid[m*5:(m+1)*5,n*5:(n+1)*5] = filt
            m += 1
            if m >= 32:
                m = 0
                n += 1
    plt.imshow(grid, cmap='gray')
    plt.title("W2")
    plt.show()


if __name__ == '__main__':
    main()
