# GMM using Variational Inference

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import multivariate_normal as mvn, dirichlet, wishart
from scipy.special import digamma, gamma

def get_cost(X, K, cluster_assignments, phi, alphas, mu_means, mu_covs, a, B, orig_alphas, orig_c, orig_a, orig_B):
  N, D = X.shape
  total = 0
  ln2pi = np.log(2*np.pi)

  # calculate B inverse since we will need it
  Binv = np.empty((K, D, D))
  for j in xrange(K):
    Binv[j] = np.linalg.inv(B[j])

  # calculate expectations first 
  Elnpi = digamma(alphas) - digamma(alphas.sum()) # E[ln(pi)]
  Elambda = np.empty((K, D, D))
  Elnlambda = np.empty(K)
  for j in xrange(K):
    Elambda[j] = a[j]*Binv[j]
    Elnlambda[j] = D*np.log(2) - np.log(np.linalg.det(B[j]))
    for d in xrange(D):
      Elnlambda[j] += digamma(a[j]/2.0 + (1 - d)/2.0)

  # now calculate the log joint likelihood
  # Gaussian part
  # total -= N*D*ln2pi
  # total += 0.5*Elnlambda.sum()
  # for j in xrange(K):
  #   # total += 0.5*Elnlambda[j] # vectorized
  #   for i in xrange(N):
  #     if cluster_assignments[i] == j:
  #       diff_ij = X[i] - mu_means[j]
  #       total -= 0.5*( diff_ij.dot(Elambda[j]).dot(diff_ij) + np.trace(Elambda[j].dot(mu_covs[j])) )

  # mixture coefficient part
  # total += Elnpi.sum()

  # use phi instead
  for j in xrange(K):
    for i in xrange(N):
      diff_ij = X[i] - mu_means[j]
      inside = Elnlambda[j] - D*ln2pi
      inside += -diff_ij.dot(Elambda[j]).dot(diff_ij) - np.trace(Elambda[j].dot(mu_covs[j]))
      # inside += Elnpi[j]
      total += phi[i,j]*(0.5*inside + Elnpi[j])
  

  # E{lnp(mu)} - based on original prior
  for j in xrange(K):
    E_mu_dot_mu = np.trace(mu_covs[j]) + mu_means[j].dot(mu_means[j])
    total += -0.5*D*np.log(2*np.pi*orig_c) - 0.5*E_mu_dot_mu/orig_c

  # print "total:", total

  # E{lnp(lambda)} - based on original prior
  for j in xrange(K):
    total += (orig_a[j] - D - 1)/2.0*Elnlambda[j] - 0.5*np.trace(orig_B[j].dot(Elambda[j]))
    # print "total 1:", total
    total += -orig_a[j]*D/2.0*np.log(2) + 0.5*orig_a[j]*np.log(np.linalg.det(orig_B[j]))
    # print "total 2:", total
    total -= D*(D-1)/4.0*np.log(np.pi)
    # print "total 3:", total
    for d in xrange(D):
      total -= np.log(gamma(orig_a[j]/2.0 + (1 - d)/2.0))

  # E{lnp(pi)} - based on original prior
  # - lnB(orig_alpha) + sum[j]{ orig_alpha[j] - 1}*E[lnpi_j]
  total += np.log(gamma(orig_alphas.sum())) - np.log(gamma(orig_alphas)).sum()
  total += ((orig_alphas - 1)*Elnpi).sum() # should be 0 since orig_alpha = 1

  # calculate entropies of the q distributions
  # q(c)
  for i in xrange(N):
    total += stats.entropy(phi[i]) # categorical entropy

  # q(pi)
  total += dirichlet.entropy(alphas)

  # q(mu)
  for j in xrange(K):
    total += mvn.entropy(cov=mu_covs[j])

  # q(lambda)
  for j in xrange(K):
    total += wishart.entropy(df=a[j], scale=Binv[j])

  return total



def gmm(X, K, max_iter=100):
  N, D = X.shape

  # parameters for pi, mu, and precision
  alphas = np.ones(K, dtype=np.float32) # prior parameter for pi (dirichlet)
  orig_alphas = np.ones(K, dtype=np.float32) # prior parameter for pi (dirichlet)
  # mu_means = np.zeros((K, D), dtype=np.float32) # prior mean for mu (normal) ### No!
  # mu_covs = np.empty((K, D, D), dtype=np.float32) # prior covariance for mu (normal)
  
  orig_c = 10.0
  # for k in xrange(K):
  #   mu_covs[k] = np.eye(D)*orig_c
    

  orig_a = np.ones(K, dtype=np.float32)*D
  a = np.ones(K, dtype=np.float32)*D # prior for precision (wishart)
  orig_B = np.empty((K, D, D))
  B = np.empty((K, D, D)) # precision (wishart)
  empirical_cov = np.cov(X.T)
  for k in xrange(K):
    B[k] = (D/10.0)*empirical_cov
    orig_B[k] = (D/10.0)*empirical_cov


  # try random init instead
  # mu_means = np.random.randn(K, D)*orig_c
  mu_means = np.empty((K, D))
  for j in xrange(K):
    mu_means[j] = X[np.random.choice(N)]
  mu_covs = wishart.rvs(df=orig_a[0], scale=np.linalg.inv(B[0]), size=K)

  costs = np.zeros(max_iter)
  for iter_idx in xrange(max_iter):
    # calculate q(c[i])
    # phi = np.empty((N,K)) # index i = sample, index j = cluster
    t1 = np.empty(K)
    t2 = np.empty((N,K))
    t3 = np.empty(K)
    t4 = np.empty(K)

    # calculate this first because we will use it multiple times
    Binv = np.empty((K, D, D))
    for j in range(K):
      Binv[j] = np.linalg.inv(B[j])

    for j in xrange(K):
      # calculate t1
      t1[j] = -np.log(np.linalg.det(B[j]))
      for d in xrange(D):
        t1[j] += digamma( (1 - d + a[j])/2.0 )

      # calculate t2
      for i in xrange(N):
        diff_ij = X[i] - mu_means[j]
        t2[i,j] = diff_ij.dot( (a[j]*Binv[j] ).dot(diff_ij) )

      # calculate t3
      t3[j] = np.trace( a[j]*Binv[j].dot(mu_covs[j]) )
      
      # calculate t4
      t4[j] = digamma(alphas[j]) - digamma(alphas.sum())

    # calculate phi from t's
    # MAKE SURE 1-d array gets added to 2-d array correctly
    phi = np.exp(0.5*t1 - 0.5*t2 - 0.5*t3 + t4)
    # print "phi before normalize:", phi
    phi = phi / phi.sum(axis=1, keepdims=True)

    # print "phi:", phi

    cluster_assignments = phi.argmax(axis=1)

    n = phi.sum(axis=0) # there should be K of these
    # print "n[j]:", n

    # update q(pi)
    alphas = orig_alphas + n
    # print "alphas:", alphas

    # update q(mu)
    for j in xrange(K):
      mu_covs[j] = np.linalg.inv( (1.0/orig_c)*np.eye(D) + n[j]*a[j]*Binv[j] )
      mu_means[j] = mu_covs[j].dot( a[j]*Binv[j] ).dot(phi[:,j].dot(X))

    # print "means:", mu_means
    # print "mu_covs:", mu_covs

    # update q(lambda)
    a = orig_a + n
    for j in xrange(K):
      B[j] = orig_B[j].copy()
      for i in xrange(N):
        diff_ij = X[i] - mu_means[j]
        B[j] += phi[i,j]*(np.outer(diff_ij, diff_ij) + mu_covs[j])

    # print "a[j]:", a
    # print "B[j]:", B

    costs[iter_idx] = get_cost(X, K, cluster_assignments, phi, alphas, mu_means, mu_covs, a, B, orig_alphas, orig_c, orig_a, orig_B)

  plt.plot(costs)
  plt.title("Costs")
  plt.show()

  print "cluster assignments:\n", cluster_assignments
  plt.scatter(X[:,0], X[:,1], c=cluster_assignments, s=100, alpha=0.7)
  plt.show()


def main():
  X = pd.read_csv('data.txt', header=None).as_matrix()

  # for K in (2,4):
  for K in (2,4,10,25):
    gmm(X, K)

if __name__ == '__main__':
  main()