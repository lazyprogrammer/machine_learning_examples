# In this file we compare the progression of the cost function vs. iteration
# for 3 cases:
# 1) full gradient descent
# 2) batch gradient descent
# 3) stochastic gradient descent
#
# We use the PCA-transformed data to keep the dimensionality down (D=300)
# I've tailored this example so that the training time for each is feasible.
# So what we are really comparing is how quickly each type of GD can converge,
# (but not actually waiting for convergence) and what the cost looks like at
# each iteration.
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

from util import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator


def main():
    Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    # 1. full
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    for i in range(50):
        p_y = forward(Xtrain, W, b)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        

        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for full GD:", datetime.now() - t0)


    # 2. stochastic
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in range(50): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for n in range(min(N, 500)): # shortcut so it won't take so long...
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_stochastic.append(ll)

        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for SGD:", datetime.now() - t0)


    # 3. batch
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N // batch_sz

    t0 = datetime.now()
    for i in range(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_batch.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for batch GD:", datetime.now() - t0)



    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()