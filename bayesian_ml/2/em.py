# expectation-maximization for the model:
# x(n) ~ N(Wz(n), sigma**2 I) (observed variables)
# z(n) ~ N(0, I) (latent variables)
# W ~ N(0, 1/lambda)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

N = 1000
lam = 1.0
sigma = 1.0
D = 2
K = 3
sigmaI = sigma * sigma * np.eye(D)

# generate the data
Z = np.random.randn(N, K)
W0 = np.random.randn(D, K)*lam
X = np.zeros((N, D))
for i in xrange(N):
  X[i] = np.random.multivariate_normal(mean=W0.dot(Z[i]), cov=sigmaI)

def loglikelihood(X, Z, W):
  ZW = Z.dot(W.T)
  LL = 0
  for i in xrange(N):
    ll = mvn.logpdf(X[i], mean=ZW[i], cov=sigmaI)
    LL += ll
  LL += norm.logpdf(W.flatten(), scale=1/lam).sum()
  return LL

# do EM
W = np.random.randn(D, K) / np.sqrt(D + K)
costs = []
for t in xrange(50):
  # E-step
  R = np.linalg.solve(W.dot(W.T) + sigmaI, W).T
  # test = W.T.dot( np.linalg.inv(W.dot(W.T) + sigmaI) )
  # print "R:", R
  # print "test:", test
  # diff = np.abs(R - test).sum()
  # print "diff:", diff
  # assert(diff < 10e-10)
  Ez = X.dot(R.T)

  # M-step
  xzT = X.T.dot(Ez)
  toinvert = N*(np.eye(K) - R.dot(W)) + Ez.T.dot(Ez) + sigma*sigma*lam*np.eye(K)
  W = np.linalg.solve(toinvert, xzT.T).T
  # test = xzT.dot( np.linalg.inv(toinvert) )
  # print "W:", W
  # print "test:", test
  # diff = np.abs(W - test).sum()
  # print "diff:", diff
  # assert(diff < 10e-5)

  # likelihood
  cost = loglikelihood(X, Ez, W)
  costs.append(cost)

plt.plot(costs)
plt.show()

print "actual W:", W0
print "predicted W:", W

print "log-likelihood given real W:", loglikelihood(X, Z, W0)

print "log-likelihood found:", costs[-1]
