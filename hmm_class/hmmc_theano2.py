# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Continuous-observation HMM in Theano using gradient descent.

# This script differs from hmmc_theano.py in the following way:
# Instead of re-normalizing the parameters at each iteration,
# we instead make the parameters free to vary between -inf to +inf.
# We then use softmax to ensure the probabilities are positive and sum to 1.

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import wave
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

# from theano.sandbox import solve # does not have gradient functionality
from generate_c import get_signals, big_init


class HMM:
    def __init__(self, M, K):
        self.M = M # number of hidden states
        self.K = K # number of Gaussians
    
    def fit(self, X, learning_rate=1e-2, max_iter=10):
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        N = len(X)
        D = X[0].shape[1] # assume each x is organized (T, D)

        pi0 = np.ones(self.M) # initial state distribution
        A0 = np.random.randn(self.M, self.M) # state transition matrix
        R0 = np.ones((self.M, self.K)) # mixture proportions
        mu0 = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                mu0[i,k] = x[random_time_idx]
        sigma0 = np.random.randn(self.M, self.K, D, D)

        thx, cost = self.set(pi0, A0, R0, mu0, sigma0)

        pi_update = self.preSoftmaxPi - learning_rate*T.grad(cost, self.preSoftmaxPi)
        A_update = self.preSoftmaxA - learning_rate*T.grad(cost, self.preSoftmaxA)
        R_update = self.preSoftmaxR - learning_rate*T.grad(cost, self.preSoftmaxR)
        mu_update = self.mu - learning_rate*T.grad(cost, self.mu)
        sigma_update = self.sigmaFactor - learning_rate*T.grad(cost, self.sigmaFactor)

        updates = [
            (self.preSoftmaxPi, pi_update),
            (self.preSoftmaxA, A_update),
            (self.preSoftmaxR, R_update),
            (self.mu, mu_update),
            (self.sigmaFactor, sigma_update),
        ]

        train_op = theano.function(
            inputs=[thx],
            updates=updates,
        )

        costs = []
        for it in range(max_iter):
            print("it:", it)
            
            for n in range(N):
                c = self.log_likelihood_multi(X).sum()
                print("c:", c)
                costs.append(c)
                train_op(X[n])

        plt.plot(costs)
        plt.show()

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, sigmaFactor):
        self.preSoftmaxPi = theano.shared(preSoftmaxPi)
        self.preSoftmaxA = theano.shared(preSoftmaxA)
        self.preSoftmaxR = theano.shared(preSoftmaxR)
        self.mu = theano.shared(mu)
        self.sigmaFactor = theano.shared(sigmaFactor)
        M, K = preSoftmaxR.shape
        self.M = M
        self.K = K

        pi = T.nnet.softmax(self.preSoftmaxPi).flatten()
        A = T.nnet.softmax(self.preSoftmaxA)
        R = T.nnet.softmax(self.preSoftmaxR)


        D = self.mu.shape[2]
        twopiD = (2*np.pi)**D

        # set up theano variables and functions
        thx = T.matrix('X') # represents a TxD matrix of sequential observations
        def mvn_pdf(x, m, S):
            k = 1 / T.sqrt(twopiD * T.nlinalg.det(S))
            e = T.exp(-0.5*(x - m).T.dot(T.nlinalg.matrix_inverse(S).dot(x - m)))
            return k*e

        def gmm_pdf(x):
            def state_pdfs(xt):
                def component_pdf(j, xt):
                    Bj_t = 0
                    # j = T.cast(j, 'int32')
                    for k in range(self.K):
                        # k = int(k)
                        # a = R[j,k]
                        # b = mu[j,k]
                        # c = sigma[j,k]
                        L = self.sigmaFactor[j,k]
                        S = L.dot(L.T)
                        Bj_t += R[j,k] * mvn_pdf(xt, self.mu[j,k], S)
                    return Bj_t

                Bt, _ = theano.scan(
                    fn=component_pdf,
                    sequences=T.arange(self.M),
                    n_steps=self.M,
                    outputs_info=None,
                    non_sequences=[xt],
                )
                return Bt

            B, _ = theano.scan(
                fn=state_pdfs,
                sequences=x,
                n_steps=x.shape[0],
                outputs_info=None,
            )
            return B.T
        
        B = gmm_pdf(thx)
        # scale = T.zeros((thx.shape[0], 1), dtype=theano.config.floatX)
        # scale[0] = (self.pi*B[:,0]).sum()

        def recurrence(t, old_a, B):
            a = old_a.dot(A) * B[:, t]
            s = a.sum()
            return (a / s), s

        [alpha, scale], _ = theano.scan(
            fn=recurrence,
            sequences=T.arange(1, thx.shape[0]),
            outputs_info=[pi*B[:,0], None],
            n_steps=thx.shape[0]-1,
            non_sequences=[B],
        )

        cost = -T.log(scale).sum()
        self.cost_op = theano.function(
            inputs=[thx],
            outputs=cost,
        )
        return thx, cost

    def log_likelihood_multi(self, X):
        return np.array([self.cost_op(x) for x in X])


def real_signal():
    spf = wave.open('helloworld.wav', 'r')

    #Extract Raw Audio from Wav File
    # If you right-click on the file and go to "Get Info", you can see:
    # sampling rate = 16000 Hz
    # bits per sample = 16
    # The first is quantization in time
    # The second is quantization in amplitude
    # We also do this for images!
    # 2^16 = 65536 is how many different sound levels we have
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()

    hmm = HMM(3, 3)
    # signal needs to be of shape N x T(n) x D
    hmm.fit(signal.reshape(1, T, 1), learning_rate=2e-7, max_iter=20)


def fake_signal():
    signals = get_signals()
    hmm = HMM(5, 3)
    hmm.fit(signals, max_iter=3)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for fitted params:", L)

    # test in actual params
    _, _, _, pi, A, R, mu, sigma = big_init()

    # turn these into their "pre-softmax" forms
    pi = np.log(pi)
    A = np.log(A)
    R = np.log(R)

    # decompose sigma using cholesky factorization
    sigma = np.linalg.cholesky(sigma)

    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for actual params:", L)

if __name__ == '__main__':
    # real_signal()
    fake_signal()

