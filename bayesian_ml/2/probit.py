# probit regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
# from scipy.stats import t
from sortedcontainers import SortedList


class ProbitRegression:
  def fit(self, X, Y, sigma=1.5, lam=1, show_w=set(), Q=None):

    # setup
    N, D = X.shape
    self.w = np.random.randn(D) / np.sqrt(D) # does not work if you don't scale first!
    Eq = np.zeros(N)
    idx1 = (Y == 1)
    idx0 = (Y == 0)
    A = lam*np.eye(D) + X.T.dot(X) / sigma**2
    costs = []

    for t in xrange(100):
      # calculate ln(p(Y, w | X))
      cdf = norm.cdf(X.dot(self.w) / sigma)
      cost = -lam/2 * self.w.dot(self.w) + Y.dot(np.log(cdf)) + (1 - Y).dot(np.log(1 - cdf))
      costs.append(cost)

      # E step
      Xw = X.dot(self.w)
      pdf = norm.pdf(-Xw / sigma)
      cdf = norm.cdf(-Xw / sigma)
      Eq[idx1] = Xw[idx1] + sigma*(pdf[idx1] / (1 - cdf[idx1]))
      Eq[idx0] = Xw[idx0] + sigma*(-pdf[idx0] / cdf[idx0])

      # M step
      b = X.T.dot(Eq) / sigma**2
      self.w = np.linalg.solve(A, b)

      if show_w and t in show_w:
        plot_image(self.w, Q, "iteration: %s" % (t+1))

        
    plt.plot(costs)
    plt.show()

    self.sigma = sigma
    self.lam = lam


  def predict_proba(self, X):
    N, D = X.shape
    return norm.cdf(X.dot(self.w) / self.sigma)

  def predict(self, X):
    return np.round(self.predict_proba(X))

  def score(self, X, Y):
    return np.mean(self.predict(X) == Y)

  def confusion_matrix(self, X, Y):
    P = self.predict(X)
    M = np.zeros((2, 2))
    M[0,0] = np.sum(P[Y == 0] == Y[Y == 0])
    M[0,1] = np.sum(P[Y == 0] != Y[Y == 0])
    M[1,0] = np.sum(P[Y == 1] != Y[Y == 1])
    M[1,1] = np.sum(P[Y == 1] == Y[Y == 1])
    return M

  def get_3_misclassified(self, X, Y):
    P = self.predict(X)
    N = len(Y)
    samples = np.random.choice(N, 3, replace=False, p=(P != Y)/float(np.sum(P != Y)))
    return X[samples], Y[samples], P[samples]

  def get_3_most_ambiguous(self, X, Y):
    P = self.predict_proba(X)
    N = len(X)
    sl = SortedList(load=3) # stores (distance, sample index) tuples
    for n in xrange(N):
      p = P[n]
      dist = np.abs(p - 0.5)
      if len(sl) < 3:
        sl.add( (dist, n) )
      else:
        if dist < sl[-1][0]:
          del sl[-1]
          sl.add( (dist, n) )
    indexes = [v for k, v in sl]
    return X[indexes], Y[indexes]

def plot_image(x, Q, title):
  im = Q.dot(x)
  plt.imshow(im.reshape(28,28), cmap='gray')
  plt.title(title)
  plt.show()

if __name__ == '__main__':
  Xtrain = pd.read_csv('Xtrain.csv', header=None).as_matrix()
  Xtest = pd.read_csv('Xtest.csv', header=None).as_matrix()
  Ytrain = pd.read_csv('ytrain.csv', header=None).as_matrix().flatten()
  # print "Ytrain.shape:", Ytrain.shape
  Ytest = pd.read_csv('ytest.csv', header=None).as_matrix().flatten()
  model = ProbitRegression()
  model.fit(Xtrain, Ytrain)
  print "train accuracy:", model.score(Xtrain, Ytrain)
  print "test accuracy:", model.score(Xtest, Ytest)

  # confusion matrix
  M = model.confusion_matrix(Xtest, Ytest)
  print "confusion matrix:"
  print M
  print "N:", len(Ytest)
  print "sum(M):", M.sum()

  # plot 3 misclassified
  Q = pd.read_csv('Q.csv', header=None).as_matrix()
  misclassified, targets, predictions = model.get_3_misclassified(Xtrain, Ytrain)
  for x, y, p in zip(misclassified, targets, predictions):
    plot_image(x, Q, 'misclassified target=%s prediction=%s' % (y, int(p)))

  # ambiguous
  ambiguous, targets = model.get_3_most_ambiguous(Xtrain, Ytrain)
  for x, y in zip(ambiguous, targets):
    plot_image(x, Q, 'ambiguous target=%s' % y)

  # show w
  model.fit(Xtrain, Ytrain, show_w=set([0, 4, 9, 24, 49, 99]), Q=Q)
