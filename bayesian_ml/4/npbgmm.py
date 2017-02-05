# GMM using Bayesian Nonparametric Clustering
# Gaussian Mixture Model
# Dirichlet Process
# Gibbs Sampling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import multivariate_normal as mvn, dirichlet, wishart
from scipy.special import digamma, gamma

# scipy wishart!!!
# parameters are df, sigma=scale
# for us, df = a, inv(sigma) = B


def marginal(x, c, m, a, B):
  D = len(x)
  k0 = ( c / (np.pi * (1 + c)) )**(D/2.0)
  k1top = np.linalg.det(B + (c/(1+c)*np.outer(x - m, x - m)))**(-(a + 1.0)/2.0)
  k1bot = np.linalg.det(B)**(-a/2.0)
  k1 = k1top/k1bot
  k2log = 0
  for d in xrange(D):
    k2log += np.log(gamma( (a+1.0)/2.0 + (1.0-d)/2.0 )) - np.log(gamma( a/2.0 + (1.0-d)/2.0 ))
  k2 = np.exp(k2log)
  return k0*k1*k2


def normalize_phi_hat(phi_hat):
  # phi_hat is a dictionary: cluster index -> non-normalized probability of that cluster
  # normalization done in place so no need to return anything
  total = np.sum(phi_hat.values())

  for j, p_hat in phi_hat.iteritems():
    phi_hat[j] = p_hat / total


def sample_cluster_identity(phi):
  # phi is a dictionary: cluster index -> probability of that cluster
  # print "dictionary sample from:", phi
  p = np.random.random()
  cumulative = 0
  for j, q in phi.iteritems():
    cumulative += q
    if p < cumulative:
      return j
  # print "cumulative:", cumulative
  assert(False) # should never get here because cumulative = 1 by now


def sample_from_prior(c0, m0, a0, B0):
  precision0 = wishart.rvs(df=a0, scale=np.linalg.inv(B0))
  cov = np.linalg.inv(precision0)
  mean = mvn.rvs(mean=m0, cov=cov/c0)
  return mean, cov


# samples mu, sigma from P(mu, sigma | X)
def sample_from_X(X, m0, c0, a0, B0):
  N = len(X)
  s = float(N)
  m = (c0 / (s + c0))*m0 + (1 / (s + c0))*X.sum(axis=0)
  c = s + c0
  a = s + a0
  meanX = X.mean(axis=0)
  B = (s / (a0*s + 1)) * np.outer(meanX - m0, meanX - m0) + B0
  for i in xrange(N):
    B += np.outer(X[i] - meanX, X[i] - meanX)
  return sample_from_prior(c, m, a, B)


def gmm(X, T=500):
  N, D = X.shape

  m0 = X.mean(axis=0)
  c0 = 0.1
  a0 = float(D)
  B0 = c0*D*np.cov(X.T)
  alpha0 = 1.0

  # cluster assignments - originally everything is assigned to cluster 0
  C = np.zeros(N)

  # keep as many as we need for each gaussian
  # originally we sample from the prior
  # TODO: just use the function above
  precision0 = wishart.rvs(df=a0, scale=np.linalg.inv(B0))
  covariances = [np.linalg.inv(precision0)]
  means = [mvn.rvs(mean=m0, cov=covariances[0]/c0)]

  cluster_counts = [1]
  K = 1
  observations_per_cluster = np.zeros((T, 6))
  for t in xrange(T):
    if t % 20 == 0:
      print t
    # 1) calculate phi[i,j]
    # Notes:
    # MANY new clusters can be made each iteration
    # A cluster can be DESTROYED if a x[i] is the only pt in cluster j and gets assigned to a new cluster
    # phi = np.empty((N, K))
    list_of_cluster_indices = range(K)
    next_cluster_index = K
    # phi = [] # TODO: do we need this at all?
    for i in xrange(N):
      phi_i = {}
      for j in list_of_cluster_indices:
        # don't loop through xrange(K) because clusters can be created or destroyed as we loop through i
        nj_noti = np.sum(C[:i] == j) + np.sum(C[i+1:] == j)
        if nj_noti > 0:
          # existing cluster
          # phi[i,j] = N(x[i] | mu[j], cov[j]) * nj_noti / (alpha0 + N - 1)
          # using the sampled mu / covs
          phi_i[j] = mvn.pdf(X[i], mean=means[j], cov=covariances[j]) * nj_noti / (alpha0 + N - 1.0)

        # new cluster
        # create a possible new cluster for every sample i
        # but only keep it if sample i occupies this new cluster j'
        # i.e. if C[i] = j' when we sample C[i]
        # phi[i,j'] = alpha0 / (alpha0 + N - 1) * p(x[i])
        # p(x[i]) is a marginal integrated over mu and precision
        phi_i[next_cluster_index] = alpha0 / (alpha0 + N - 1.0) * marginal(X[i], c0, m0, a0, B0)

      # normalize phi[i] and assign C[i] to its new cluster by sampling from phi[i]
      normalize_phi_hat(phi_i)

      # if C[i] = j' (new cluster), generate mu[j'] and cov[j']
      C[i] = sample_cluster_identity(phi_i)
      if C[i] == next_cluster_index:
        list_of_cluster_indices.append(next_cluster_index)
        next_cluster_index += 1
        new_mean, new_cov = sample_from_prior(c0, m0, a0, B0)
        means.append(new_mean)
        covariances.append(new_cov)

      # destroy any cluster with no points in it
      clusters_to_remove = []
      tot = 0
      for j in list_of_cluster_indices:
        nj = np.sum(C == j)
        # print "number of pts in cluster %d:" % j, nj
        tot += nj
        if nj == 0:
          clusters_to_remove.append(j)
      # print "tot:", tot
      assert(tot == N)
      for j in clusters_to_remove:
        list_of_cluster_indices.remove(j)

    # DEBUG - make sure no clusters are empty
    # counts = [np.sum(C == j) for j in list_of_cluster_indices]
    # for c in counts:
    #   assert(c > 0)

    # re-order the cluster indexes so they range from 0..new K - 1
    new_C = np.zeros(N)
    for new_j in xrange(len(list_of_cluster_indices)):
      old_j = list_of_cluster_indices[new_j]
      new_C[C == old_j] = new_j
    C = new_C
    K = len(list_of_cluster_indices)
    list_of_cluster_indices = range(K) # redundant but if removed will break counts

    cluster_counts.append(K)
    # 2) calculate the new mu, covariance for every currently non-empty cluster
    # i.e. SAMPLE mu, cov from the new cluster assignments
    means = []
    covariances = []
    for j in xrange(K):
      # first calculate m', c', a', B'
      # then call the function that samples a mean and covariance using these
      mean, cov = sample_from_X(X[C == j], m0, c0, a0, B0)
      means.append(mean)
      covariances.append(cov)

    # plot number of observations per cluster for 6 most probable clusters per iteration
    counts = sorted([np.sum(C == j) for j in list_of_cluster_indices], reverse=True)
    # print "counts:", counts
    if len(counts) < 6:
      observations_per_cluster[t,:len(counts)] = counts
    else:
      observations_per_cluster[t] = counts[:6]

  # plot number of clusters per iteration
  plt.plot(cluster_counts)
  plt.show()

  # plot number of observations per cluster for 6 most probable clusters per iteration
  plt.plot(observations_per_cluster)
  plt.show()


def main():
  X = pd.read_csv('data.txt', header=None).as_matrix()
  gmm(X)

if __name__ == '__main__':
  main()
