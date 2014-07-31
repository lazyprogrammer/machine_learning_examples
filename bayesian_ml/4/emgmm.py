# GMM using Expectation-Maximization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def gmm(X, K, max_iter=20, smoothing=1e-2):
  N, D = X.shape
  M = np.zeros((K, D))
  R = np.zeros((N, K))
  C = np.zeros((K, D, D))
  pi = np.ones(K) / K # uniform

  # initialize M to random, initialize C to spherical with variance 1
  for k in xrange(K):
    M[k] = X[np.random.choice(N)]
    C[k] = np.eye(D)

  costs = np.zeros(max_iter)
  weighted_pdfs = np.zeros((N, K)) # we'll use these to store the PDF value of sample n and Gaussian k
  for i in xrange(max_iter):
    # step 1: determine assignments / resposibilities
    for k in xrange(K):
      weighted_pdfs[:,k] = pi[k]*multivariate_normal.pdf(X, M[k], C[k])

    R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
    # for k in xrange(K):
    #   for n in xrange(N):
    #     R[n,k] = weighted_pdfs[n,k] / weighted_pdfs[n,:].sum()

    # step 2: recalculate params
    for k in xrange(K):
      Nk = R[:,k].sum()
      pi[k] = Nk / N
      M[k] = R[:,k].dot(X) / Nk
      C[k] = np.sum(R[n,k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in xrange(N)) / Nk + np.eye(D)*smoothing


    costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
    if i > 0:
      if np.abs(costs[i] - costs[i-1]) < 0.1:
        break

  plt.plot(costs)
  plt.title("Costs")
  plt.show()

  random_colors = np.random.random((K, 3))
  # colors = R.dot(random_colors)
  # plt.scatter(X[:,0], X[:,1], c=colors)
  plt.scatter(X[:,0], X[:,1], c=R.argmax(axis=1))
  plt.show()

  print "pi:", pi
  print "means:", M
  print "covariances:", C
  return R


def main():
  X = pd.read_csv('data.txt', header=None).as_matrix()

  # what does it look like without clustering?
  plt.scatter(X[:,0], X[:,1])
  plt.show()

  for K in (2,4,8,10):
    gmm(X, K, max_iter=100, smoothing=0)



if __name__ == '__main__':
  main()