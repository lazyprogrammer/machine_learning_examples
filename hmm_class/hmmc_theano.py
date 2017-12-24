# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Continuous-observation HMM in Theano using gradient descent.
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

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M, K):
        self.M = M # number of hidden states
        self.K = K # number of Gaussians
    
    def fit(self, X, learning_rate=1e-2, max_iter=10):
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        N = len(X)
        D = X[0].shape[1] # assume each x is organized (T, D)

        pi0 = np.ones(self.M) / self.M # initial state distribution
        A0 = random_normalized(self.M, self.M) # state transition matrix
        R0 = np.ones((self.M, self.K)) / self.K # mixture proportions
        mu0 = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                mu0[i,k] = x[random_time_idx]
        sigma0 = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                sigma0[j,k] = np.eye(D)

        thx, cost = self.set(pi0, A0, R0, mu0, sigma0)

        pi_update = self.pi - learning_rate*T.grad(cost, self.pi)
        pi_update = pi_update / pi_update.sum()

        A_update = self.A - learning_rate*T.grad(cost, self.A)
        A_update = A_update / A_update.sum(axis=1).dimshuffle(0, 'x')

        R_update = self.R - learning_rate*T.grad(cost, self.R)
        R_update = R_update / R_update.sum(axis=1).dimshuffle(0, 'x')

        updates = [
            (self.pi, pi_update),
            (self.A, A_update),
            (self.R, R_update),
            (self.mu, self.mu - learning_rate*T.grad(cost, self.mu)),
            (self.sigma, self.sigma - learning_rate*T.grad(cost, self.sigma)),
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

        print("A:", self.A.get_value())
        print("mu:", self.mu.get_value())
        print("sigma:", self.sigma.get_value())
        print("R:", self.R.get_value())
        print("pi:", self.pi.get_value())

        plt.plot(costs)
        plt.show()

    def set(self, pi, A, R, mu, sigma):
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.R = theano.shared(R)
        self.mu = theano.shared(mu)
        self.sigma = theano.shared(sigma)
        M, K = R.shape
        self.M = M
        self.K = K

        D = self.mu.shape[2]
        twopiD = (2*np.pi)**D

        # set up theano variables and functions
        thx = T.matrix('X') # represents a TxD matrix of sequential observations
        def mvn_pdf(x, mu, sigma):
            k = 1 / T.sqrt(twopiD * T.nlinalg.det(sigma))
            e = T.exp(-0.5*(x - mu).T.dot(T.nlinalg.matrix_inverse(sigma).dot(x - mu)))
            return k*e

        def gmm_pdf(x):
            def state_pdfs(xt):
                def component_pdf(j, xt):
                    Bj_t = 0
                    for k in range(self.K):
                        Bj_t += self.R[j,k] * mvn_pdf(xt, self.mu[j,k], self.sigma[j,k])
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
            a = old_a.dot(self.A) * B[:, t]
            s = a.sum()
            return (a / s), s

        [alpha, scale], _ = theano.scan(
            fn=recurrence,
            sequences=T.arange(1, thx.shape[0]),
            outputs_info=[self.pi*B[:,0], None],
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

    hmm = HMM(5, 3)
    # signal needs to be of shape N x T(n) x D
    hmm.fit(signal.reshape(1, T, 1), learning_rate=1e-5, max_iter=20)


def fake_signal():
    signals = get_signals()
    hmm = HMM(5, 3)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for fitted params:", L)

    # test in actual params
    _, _, _, pi, A, R, mu, sigma = big_init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for actual params:", L)

if __name__ == '__main__':
    # real_signal()
    fake_signal()

