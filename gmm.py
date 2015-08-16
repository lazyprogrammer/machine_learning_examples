import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM


def generate_data(pi, mu, sigma, N):
    # pi should be K x 1
    # mu should be K x D
    # sigma should be K x D x D
    K,D = mu.shape
    X = np.zeros((N,D))
    C = np.random.random(N)
    colors = np.zeros(N)

    pi_cdf = np.zeros(len(pi))
    pi_cdf[0] = pi[0]
    for i,p in enumerate(pi[1:]):
        pi_cdf[i+1] = pi_cdf[i] + p

    for i in xrange(N):
        k = -1
        for j,P in enumerate(pi_cdf):
            if C[i] < P:
                k = j
                colors[i] = j
                break
        X[i] = np.random.multivariate_normal(mu[k], sigma[k])
    return X, colors


def pdf(x,m,S):
    D = len(x)
    diff = x - m
    c = ((2*np.pi)**(-D/2.0)) * np.sqrt(np.linalg.det(S))
    inv = np.linalg.inv(S)
    e = np.exp(-0.5*np.dot( np.dot(diff, inv), diff ))
    if np.any(np.isnan(S)):
        print "S:", S
        raise Exception("NAN in S")
    if np.any(np.isnan(c)):
        print "S:", S
        raise Exception("NAN in c")
    if np.any(np.isnan(e)):
        raise Exception("NAN in e")
    return c*e


def expectation_maximization(X, K, max_iter=100):
    # return guess for pi, mu, sigma
    N,D = X.shape

    # initialize pi, mu, sigma
    pi = np.ones(K)/K
    #mu = np.random.randn(K,D)
    mu = np.array([[1,1], [-1,-1], [-1,1]])*3.0
    I = np.eye(D)
    sigma = np.zeros((K,D,D))
    for k in xrange(K):
        sigma[k] = I

    phi = np.zeros((N,K))
    piP = np.zeros((N,K))
    L = np.zeros(max_iter)
    for t in xrange(max_iter):
        print "t:", t

        # E step: populate phi
        for i in xrange(N):
            for k in xrange(K):
                piP[i,k] = pi[k]*pdf(X[i], mu[k], sigma[k])
        for i in xrange(N):
            phi[i] = piP[i] / piP[i].sum()

        if np.any(np.isnan(phi)):
            raise Exception("NAN in phi")

        # log the objective function
        L[t] = np.log(piP[i].sum()).sum()

        # M step
        for k in xrange(K):
            nk = phi[:,k].sum()
            pi[k] = nk / N
            mu[k] = np.dot(phi[:,k], X) / nk
            for i in xrange(N):
                diff = X[i] - mu[k]
                sigma[k] += phi[i,k]*np.outer(diff, diff)
            sigma[k] /= nk

        if np.any(np.isnan(pi)):
            raise Exception("NAN in pi")
        if np.any(np.isnan(mu)):
            raise Exception("NAN in mu")
        if np.any(np.isnan(sigma)):
            raise Exception("NAN in sigma")
        
    return pi, mu, sigma, L


def main():
    pi = np.array([0.3, 0.5, 0.2])
    mu = np.array([[1,1], [-1,-1], [-1,1]])*3
    sigma = np.array([
        [[1,0], [0,1]],
        [[2,0], [0,2]],
        [[0.5,0], [0, 0.5]],
    ])
    X, C = generate_data(pi, mu, sigma, 1000)
    plt.scatter(X[:,0], X[:,1], c=C, s=100, alpha=0.5)
    plt.show()


    # sklearn
    gmm = GMM(n_components=3, covariance_type='full')
    gmm.fit(X)
    print "pi:", gmm.weights_
    print "mu:", gmm.means_
    print "sigma:", gmm.covars_

    pi2, mu2, sigma2, L = expectation_maximization(X, len(pi))
    print "pi:", pi2
    print "mu:", mu2
    print "sigma:", sigma2
    plt.plot(L)
    plt.show()


if __name__ == '__main__':
    main()