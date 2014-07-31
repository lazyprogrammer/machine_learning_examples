# Naive Bayes with prior on mean and precision of Gaussian
# mean | precision ~ N(0, c / precision)
# precision ~ Gamma(a, b)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from sortedcontainers import SortedList


class NB:
  def fit(self, X, Y):
    self.pyy = []
    self.tinfo = []
    N, D = X.shape
    for c in (0, 1):
      pyy_c = (1.0 + np.sum(Y == c)) / (N + 1.0 + 1.0)
      self.pyy.append(pyy_c)

      # for each dimension, we need to store the data we need to calculate
      # the posterior predictive distribution
      # t-distribution with 3 params: df, center, scale
      Xc = X[Y == c]
      tinfo_c = []
      for d in xrange(D):
        # first calculate the parameters of the normal gamma
        xbar = Xc[:,d].mean()
        mu = N*xbar / (1.0 + N)
        precision = 1.0 + N
        alpha = 1.0 + N/2.0
        beta = 1.0 + 0.5*Xc[:,d].var()*N + 0.5*N*(xbar*xbar)/precision

        tinfo_cd = {
          'df': 2*alpha,
          'center': mu,
          'scale': np.sqrt( beta*(precision + 1)/(alpha * precision) ),
        }
        tinfo_c.append(tinfo_cd)
      self.tinfo.append(tinfo_c)

  def predict_proba(self, X):
    N, D = X.shape
    # P = np.zeros(N)
    # for n in xrange(N):
    #   x = X[n]

    #   pyx = []
    #   for c in (0, 1):
    #     pycx = self.pyy[c]
    #     for d in xrange(D):
    #       tinfo_cd = self.tinfo[c][d]
    #       pdf_d = t.pdf(x[d], df=tinfo_cd['df'], loc=tinfo_cd['center'], scale=tinfo_cd['scale'])
    #       pycx *= pdf_d
    #     pyx.append(pycx)

    #   py1x = pyx[1] / (pyx[0] + pyx[1])
    #   # print "p(y=1|x):", py1x
    #   P[n] = py1x

    posteriors = np.zeros((N, 2))
    for c in (0, 1):
      probability_matrix = np.zeros((N, D))
      for d in xrange(D):
        tinfo_cd = self.tinfo[c][d]
        pdf_d = t.pdf(X[:,d], df=tinfo_cd['df'], loc=tinfo_cd['center'], scale=tinfo_cd['scale'])
        probability_matrix[:,d] = pdf_d
      posteriors_c = np.prod(probability_matrix, axis=1)*self.pyy[c]
      posteriors[:,c] = posteriors_c
    P = posteriors[:,1] / np.sum(posteriors, axis=1)
    return P

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
  model = NB()
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
