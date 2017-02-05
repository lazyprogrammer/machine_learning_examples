# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Continuous-observation HMM with no scaling, but treats multiple observations
# as a concatenated single observation
import wave
import numpy as np
import matplotlib.pyplot as plt

from generate_c import get_signals, big_init, simple_init
from scipy.stats import multivariate_normal as mvn

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M, K):
        self.M = M # number of hidden states
        self.K = K # number of Gaussians
    
    def fit(self, X, max_iter=30, eps=10e-1):
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        N = len(X)
        D = X[0].shape[1] # assume each x is organized (T, D)

        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.R = np.ones((self.M, self.K)) / self.K # mixture proportions
        print "initial A:", self.A
        print "initial R:", self.R
        self.mu = np.zeros((self.M, self.K, D))
        for i in xrange(self.M):
            for k in xrange(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                self.mu[i,k] = x[random_time_idx]
        self.sigma = np.zeros((self.M, self.K, D, D))
        for j in xrange(self.M):
            for k in xrange(self.K):
                self.sigma[j,k] = np.eye(D)

        costs = []
        for it in xrange(max_iter):
            if it % 1 == 0:
                print "it:", it
            alphas = []
            betas = []
            gammas = []
            Bs = []
            # components = []
            P = np.zeros(N)

            for n in xrange(N):
                x = X[n]
                T = len(x)

                # calculate B so we can lookup when updating alpha and beta
                B = np.zeros((self.M, T))
                component = np.zeros((self.M, self.K, T)) # we'll need these later
                for j in xrange(self.M):
                    for t in xrange(T):
                        for k in xrange(self.K):
                            # print "sigma:", self.sigma[j,k]
                            p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                            component[j,k,t] = p
                            B[j,t] += p
                Bs.append(B)

                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi*B[:,0]
                for t in xrange(1, T):
                    alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
                P[n] = alpha[-1].sum()
                assert(P[n] <= 1)
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in xrange(T - 2, -1, -1):
                    beta[t] = self.A.dot(B[:,t+1] * beta[t+1])
                betas.append(beta)

                # update for Gaussians
                gamma = np.zeros((T, self.M, self.K))
                for t in xrange(T):
                    alphabeta = (alphas[n][t,:] * betas[n][t,:]).sum()
                    # print "alpabeta:", alphabeta
                    for j in xrange(self.M):
                        factor = alphas[n][t,j] * betas[n][t,j] / alphabeta
                        # mixture_j = component[j,:,t].sum()
                        for k in xrange(self.K):
                            gamma[t,j,k] = factor * component[j,k,t] / B[j,t]
                            # assert(np.abs(mixture_j - B[j,t]).sum() < 10e-10)
                gammas.append(gamma)

            cost = np.log(P).sum()
            costs.append(cost)

            # now re-estimate pi, A, R, mu, sigma
            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in xrange(N)) / N

            a_den = np.zeros((self.M, 1))
            a_num = 0
            r_num = np.zeros((self.M, self.K))
            r_den = np.zeros(self.M)
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D, D))
            for n in xrange(N):
                x = X[n]
                T = len(x)
                B = Bs[n]
                gamma = gammas[n]

                # denominator for A
                a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]

                # numerator for A
                a_num_n = np.zeros((self.M, self.M))
                for i in xrange(self.M):
                    for j in xrange(self.M):
                        for t in xrange(T-1):
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * B[j,t+1] * betas[n][t+1,j]
                a_num += a_num_n / P[n]


                # update mixture components
                r_num_n = np.zeros((self.M, self.K))
                r_den_n = np.zeros(self.M)
                for j in xrange(self.M):
                    for k in xrange(self.K):
                        for t in xrange(T):
                            r_num_n[j,k] += gamma[t,j,k]
                            r_den_n[j] += gamma[t,j,k]
                r_num += r_num_n / P[n]
                r_den += r_den_n / P[n]

                mu_num_n = np.zeros((self.M, self.K, D))
                sigma_num_n = np.zeros((self.M, self.K, D, D))
                for j in xrange(self.M):
                    for k in xrange(self.K):
                        for t in xrange(T):
                            # update means
                            mu_num_n[j,k] += gamma[t,j,k] * x[t]

                            # update covariances
                            sigma_num_n[j,k] += gamma[t,j,k] * np.outer(x[t] - self.mu[j,k], x[t] - self.mu[j,k])
                mu_num += mu_num_n / P[n]
                sigma_num += sigma_num_n / P[n]

            self.A = a_num / a_den
            # print self.A
            assert(np.all(self.A <= 1))

            # update R, mu, sigma
            for j in xrange(self.M):
                for k in xrange(self.K):
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k]

        print "A:", self.A
        print "mu:", self.mu
        print "sigma:", self.sigma
        print "R:", self.R
        print "pi:", self.pi

        plt.plot(costs)
        plt.show()

    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        alpha = np.zeros((T, self.M))

        B = np.zeros((self.M, T))
        for j in xrange(self.M):
            for t in xrange(T):
                for k in xrange(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        alpha[0] = self.pi*B[:,0]
        for t in xrange(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
        return alpha[-1].sum()

    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

    def set(self, pi, A, R, mu, sigma):
        self.pi = pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma
        M, K = R.shape
        self.M = M
        self.K = K


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

    hmm = HMM(10)
    hmm.fit(signal.reshape(1, T))


def fake_signal(init=simple_init):
    signals = get_signals(N=10, T=10, init=init)
    # for signal in signals:
    #     for d in xrange(signal.shape[1]):
    #         plt.plot(signal[:,d])
    # plt.show()

    hmm = HMM(2, 2)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
    print "LL for fitted params:", L

    # test in actual params
    _, _, _, pi, A, R, mu, sigma = init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print "LL for actual params:", L

if __name__ == '__main__':
    # real_signal() # will break
    fake_signal(init=simple_init)
    # fake_signal(init=big_init) # will break

