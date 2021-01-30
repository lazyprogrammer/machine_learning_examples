# In this file we compare the progression of the cost function vs. iteration
# for 3 cases:
# 1) full gradient descent
# 2) mini-batch gradient descent
# 3) stochastic gradient descent
#
# For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_normalized_data, forward, error_rate, cost, gradW, gradb, y2indicator


def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    # 1. full
    W = np.random.randn(D, 10) / np.sqrt(D)
    W0 = W.copy() # save for later
    b = np.zeros(10)
    test_losses_full = []
    lr = 0.9
    reg = 0.
    t0 = datetime.now()
    last_dt = 0
    intervals = []
    for i in range(50):
        p_y = forward(Xtrain, W, b)

        gW = gradW(Ytrain_ind, p_y, Xtrain) / N
        gb = gradb(Ytrain_ind, p_y) / N

        W += lr*(gW - reg*W)
        b += lr*(gb - reg*b)

        p_y_test = forward(Xtest, W, b)
        test_loss = cost(p_y_test, Ytest_ind)
        dt = (datetime.now() - t0).total_seconds()

        # save these
        dt2 = dt - last_dt
        last_dt = dt
        intervals.append(dt2)

        test_losses_full.append([dt, test_loss])
        if (i + 1) % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i + 1, test_loss))
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for full GD:", datetime.now() - t0)

    # save the max time so we don't surpass it in subsequent iterations
    max_dt = dt
    avg_interval_dt = np.mean(intervals)


    # 2. stochastic
    W = W0.copy()
    b = np.zeros(10)
    test_losses_sgd = []
    lr = 0.001
    reg = 0.

    t0 = datetime.now()
    last_dt_calculated_loss = 0
    done = False
    for i in range(50): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for n in range(N):
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            p_y = forward(x, W, b)

            gW = gradW(y, p_y, x)
            gb = gradb(y, p_y)

            W += lr*(gW - reg*W)
            b += lr*(gb - reg*b)

            dt = (datetime.now() - t0).total_seconds()
            dt2 = dt - last_dt_calculated_loss

            if dt2 > avg_interval_dt:
                last_dt_calculated_loss = dt
                p_y_test = forward(Xtest, W, b)
                test_loss = cost(p_y_test, Ytest_ind)
                test_losses_sgd.append([dt, test_loss])

            # time to quit
            if dt > max_dt:
                done = True
                break
        if done:
            break

        if (i + 1) % 1 == 0:
            print("Cost at iteration %d: %.6f" % (i + 1, test_loss))
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for SGD:", datetime.now() - t0)


    # 3. mini-batch
    W = W0.copy()
    b = np.zeros(10)
    test_losses_batch = []
    batch_sz = 500
    lr = 0.08
    reg = 0.
    n_batches = int(np.ceil(N / batch_sz))


    t0 = datetime.now()
    last_dt_calculated_loss = 0
    done = False
    for i in range(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j + 1)*batch_sz,:]
            y = tmpY[j*batch_sz:(j + 1)*batch_sz,:]
            p_y = forward(x, W, b)

            current_batch_sz = len(x)
            gW = gradW(y, p_y, x) / current_batch_sz
            gb = gradb(y, p_y) / current_batch_sz

            W += lr*(gW - reg*W)
            b += lr*(gb - reg*b)

            dt = (datetime.now() - t0).total_seconds()
            dt2 = dt - last_dt_calculated_loss

            if dt2 > avg_interval_dt:
                last_dt_calculated_loss = dt
                p_y_test = forward(Xtest, W, b)
                test_loss = cost(p_y_test, Ytest_ind)
                test_losses_batch.append([dt, test_loss])

            # time to quit
            if dt > max_dt:
                done = True
                break
        if done:
            break

        if (i + 1) % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i + 1, test_loss))
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for mini-batch GD:", datetime.now() - t0)


    # convert to numpy arrays
    test_losses_full = np.array(test_losses_full)
    test_losses_sgd = np.array(test_losses_sgd)
    test_losses_batch = np.array(test_losses_batch)

    
    plt.plot(test_losses_full[:,0], test_losses_full[:,1], label="full")
    plt.plot(test_losses_sgd[:,0], test_losses_sgd[:,1], label="sgd")
    plt.plot(test_losses_batch[:,0], test_losses_batch[:,1], label="mini-batch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()