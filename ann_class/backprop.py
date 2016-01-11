# backpropagation example for deep learning in python class.
#
# the notes for this class can be found at: 
# https://www.udemy.com/data-science-deep-learning-in-python

import numpy as np

def forward(X, W1, W2):
    Z = 1 / (1 + np.exp(-X.dot(W1)))
    A = Z.dot(W2)
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1] # H is (N, M)

    # # slow
    # ret1 = np.zeros((M, K))
    # for n in xrange(N):
    #     for m in xrange(M):
    #         for k in xrange(K):
    #             ret1[m,k] += T[n,k]*(1 - Y[n,k])*Z[n,m]

    # # a bit faster - let's not loop over m
    # ret2 = np.zeros((M, K))
    # for n in xrange(N):
    #     for k in xrange(K):
    #         ret2[:,k] += T[n,k]*(1 - Y[n,k])*Z[n,:]

    # assert(np.abs(ret1 - ret2).sum() < 0.00001)

    # # even faster  - let's not loop over k either
    # ret3 = np.zeros((M, K))
    # for n in xrange(N): # slow way first
    #     ret3 += np.outer( Z[n], T[n]*(1 - Y[n]) )

    # assert(np.abs(ret1 - ret3).sum() < 0.00001)

    # fastest - let's not loop over anything
    ret4 = Z.T.dot(T - Y)
    # assert(np.abs(ret1 - ret4).sum() < 0.00001)

    return ret4


def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    # slow way first
    # ret1 = np.zeros((X.shape[1], M))
    # for n in xrange(N):
    #     for k in xrange(K):
    #         for m in xrange(M):
    #             for d in xrange(D):
    #                 ret1[d,m] += T[n,k]*(1 - Y[n,k])*W2[m,k]*Z[n,m]*(1 - Z[n,m])*X[n,d]

    # fastest
    ret2 = X.T.dot( ( ( T-Y ).dot(W2.T) * ( Z*(1 - Z) ) ) )

    # assert(np.abs(ret1 - ret2).sum() < 0.00001)

    return ret2


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def main():
    X = np.random.randn(10,3)
    W1 = np.random.randn(3,5)
    W2 = np.random.randn(5,3)
    T = np.zeros((10,3))
    for i in xrange(10):
        T[ np.random.randint(3) ] = 1

    learning_rate = 0.00001
    reg = 0.1
    for epoch in xrange(100000):
        output, hidden = forward(X, W1, W2)
        if epoch % 100 == 0:
            print cost(T, output)
        W2 += learning_rate * (derivative_w2(hidden, T, output) - reg * W2)
        W1 += learning_rate * (derivative_w1(X, hidden, T, output, W2) - reg * W1)



if __name__ == '__main__':
    main()

