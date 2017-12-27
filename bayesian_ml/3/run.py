# variational-inference for linear regression
# y(i) ~ N( x(i).dot(w), 1/lambda )
# w ~ N( 0, diag(alpha_1, alpha_2, ..., alpha_D)^-1 )
# alpha_i ~ Gamma(a, b)
# lambda ~ Gamma(e, f)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma as gamma_dist
from scipy.special import gamma, digamma

def e_ln_q_gamma(a, b):
  return np.log(b) - a - np.log(np.abs(gamma(a))) + (a - 1)*digamma(a)

def objective(X, Y, C, mu, a, b, e, f, a0, b0, e0, f0):
  log2pi = np.log(2*np.pi)
  N, D = X.shape

  # E(lnX) = digamma(a) - ln(b) for X ~ Gamma(a,b)
  E_ln_lambda = digamma(e) - np.log(f)
  E_ln_alpha = digamma(a) - np.log(b)

  # model likelihood
  total = (N/2.0)*(E_ln_lambda - log2pi)
  data_total = 0
  for i in xrange(N):
    delta = Y[i] - X[i].dot(mu)
    data_total += delta*delta + X[i].dot(C).dot(X[i])
  total -= (float(e)/f)/2.0 * data_total

  # print "total after model likelihood:", total

  # w likelihood
  total -= (D/2.0)*log2pi
  for k in xrange(D):
    total += 0.5*(E_ln_alpha[k] - (float(a[k])/b[k])*(C[k,k] + mu[k]*mu[k]))

  # print "total after w likelihood:", total

  # lambda likelihood
  total += e0*np.log(f0) - np.log(gamma(e0)) + (e0 - 1)*E_ln_lambda - f0*(float(e)/f)

  # print "total after lambda likelihood:", total

  # alpha likelihood
  for k in xrange(D):
    total += a0*np.log(b0) - np.log(gamma(a0)) + (a0 - 1)*E_ln_alpha[k] - b0*(float(a[k])/b[k])

  # print "total after alpha likelihood:", total

  # entropy
  # TODO: calculate this manually
  # total -= mvn.entropy(mean=mu, cov=C)
  # e1 = mvn.entropy(cov=C)
  # e2 = 0.5*np.log( np.linalg.det(2*np.pi*np.e*C) )
  # print "e1:", e1, "e2:", e2
  # total += 0.5*np.log( np.linalg.det(2*np.pi*np.e*C) )

  total += mvn.entropy(cov=C)
  # print "det(C):", np.linalg.det(C)
  # print "total after lnq(w):", total

  # total -= gamma_dist.entropy(e, scale=1.0/f)
  # e3 = gamma_dist.entropy(e, scale=1.0/f)
  # e4 = -e_ln_q_gamma(e, f)
  # print "e3:", e3, "e4:", e4
  # assert(np.abs(e3 - e4) < 1e-8)
  total += gamma_dist.entropy(e, scale=1.0/f)
  # total -= e_ln_q_gamma(e, f)
  # print "total after lnq(lambda):", total
  for k in xrange(D):
    # total -= e_ln_q_gamma(a[k], b[k])
    total += gamma_dist.entropy(a[k], scale=1.0/b[k])
  return total


def run(num=1, T=500):
  X = pd.read_csv('X_set%s.csv' % num, header=None).as_matrix()
  Y = pd.read_csv('y_set%s.csv' % num, header=None).as_matrix().flatten()
  Z = pd.read_csv('z_set%s.csv' % num, header=None).as_matrix().flatten()
  N, D = X.shape
  print X.shape, Y.shape, Z.shape

  a0 = 1e-16
  b0 = 1e-16
  e0 = 1
  f0 = 1

  # params for q(w) - doesn't matter what we set it to, we'll update this first
  C = np.eye(D)
  mu = np.zeros(D)

  # params for q(lambda)
  e = e0
  f = f0

  # params for q(alpha)
  a = np.ones(D)*a0
  b = np.ones(D)*b0
  a0ones = np.ones(D)*a0

  # objective
  L = np.empty(T)

  for t in xrange(T):
    # update q(w)
    C = np.linalg.inv(np.diag(1.0*a/b) + (1.0*e/f)*X.T.dot(X))
    mu = C.dot((1.0*e/f)*X.T.dot(Y))

    # update q(alpha)
    a = a0ones + 0.5
    b = b0 + 0.5*(np.diag(C) + mu*mu)
    # for k in xrange(D):
    #   a[k] = a0 + 0.5
    #   b[k] = b0 + 0.5*(C[k,k] + mu[k]*mu[k])

    # update q(lambda)
    e = e0 + N/2.0
    sum_for_f = 0
    # for i in xrange(N):
    #   delta = Y[i] - X[i].dot(mu)
    #   sum_for_f += delta*delta + X[i].dot(C).dot(X[i])
    delta = Y - X.dot(mu)
    sum_for_f = delta.dot(delta) + np.trace(X.dot(C).dot(X.T))
    f = f0 + 0.5*sum_for_f

    # update L
    L[t] = objective(X, Y, C, mu, a, b, e, f, a0, b0, e0, f0)
    if t % 20 == 0:
      print "t:", t
      if num == 3:
        print "L:", L[t]

  # plot 1/E[alpha]
  plt.plot(b/a)
  plt.show()

  # 1/E[lambda]
  print "1/E[lambda]:", f/e

  # plot L
  plt.plot(L)
  plt.show()

  Yhat = X.dot(mu)
  plt.plot(Z, Yhat)
  plt.scatter(Z, Y)
  plt.plot(Z, 10*np.sinc(Z))
  plt.show()


run(1)
run(2)
run(3)