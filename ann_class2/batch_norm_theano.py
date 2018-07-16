from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util import get_normalized_data


def init_weight(M1, M2):
  return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)


class HiddenLayerBatchNorm(object):
  def __init__(self, M1, M2, f):
    self.M1 = M1
    self.M2 = M2
    self.f = f

    W = init_weight(M1, M2)
    gamma = np.ones(M2)
    beta = np.zeros(M2)

    self.W = theano.shared(W)
    self.gamma = theano.shared(gamma)
    self.beta = theano.shared(beta)

    self.params = [self.W, self.gamma, self.beta]

    # for test time
    # self.running_mean = T.zeros(M2)
    # self.running_var = T.zeros(M2)
    self.running_mean = theano.shared(np.zeros(M2))
    self.running_var = theano.shared(np.zeros(M2))

  def forward(self, X, is_training):
    activation = X.dot(self.W)
    if is_training:
      # returns:
      #   batch-normalized output
      #   batch mean
      #   batch variance
      #   running mean (for later use as population mean estimate)
      #   running var (for later use as population var estimate)
      out, batch_mean, batch_invstd, new_running_mean, new_running_var = batch_normalization_train(
        activation,
        self.gamma,
        self.beta,
        running_mean=self.running_mean,
        running_var=self.running_var,
      )

      self.running_update = [
        (self.running_mean, new_running_mean),
        (self.running_var, new_running_var),
      ]

      # if you don't trust the built-in bn function
      # batch_var = 1 / (batch_invstd * batch_invstd)
      # self.running_update = [
      #   (self.running_mean, 0.9*self.running_mean + 0.1*batch_mean),
      #   (self.running_var, 0.9*self.running_var + 0.1*batch_var),
      # ]

    else:
      out = batch_normalization_test(
        activation,
        self.gamma,
        self.beta,
        self.running_mean,
        self.running_var
      )
    return self.f(out)


class HiddenLayer(object):
  def __init__(self, M1, M2, f):
    self.M1 = M1
    self.M2 = M2
    self.f = f
    W = init_weight(M1, M2)
    b = np.zeros(M2)
    self.W = theano.shared(W)
    self.b = theano.shared(b)
    self.params = [self.W, self.b]

  def forward(self, X):
    return self.f(X.dot(self.W) + self.b)


class ANN(object):
  def __init__(self, hidden_layer_sizes):
    self.hidden_layer_sizes = hidden_layer_sizes

  def fit(self, X, Y, Xtest, Ytest, activation=T.nnet.relu, learning_rate=1e-2, mu=0.9, epochs=15, batch_sz=100, print_period=100, show_fig=True):
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)

    # initialize hidden layers
    N, D = X.shape
    self.layers = []
    M1 = D
    for M2 in self.hidden_layer_sizes:
      h = HiddenLayerBatchNorm(M1, M2, activation)
      self.layers.append(h)
      M1 = M2
      
    # final layer
    K = len(set(Y))
    h = HiddenLayer(M1, K, T.nnet.softmax)
    self.layers.append(h)

    if batch_sz is None:
      batch_sz = N

    # collect params for later use
    self.params = []
    for h in self.layers:
      self.params += h.params

    # for momentum
    dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]

    # note! we will need to build the output differently
    # for train and test (prediction)

    # set up theano functions and variables
    thX = T.matrix('X')
    thY = T.ivector('Y')

    # for training
    p_y_given_x = self.forward(thX, is_training=True)

    cost = -T.mean(T.log(p_y_given_x[T.arange(thY.shape[0]), thY]))
    prediction = T.argmax(p_y_given_x, axis=1)
    grads = T.grad(cost, self.params)

    # momentum only
    updates = [
      (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
    ] + [
      (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
    ]
    for layer in self.layers[:-1]:
      updates += layer.running_update

    train_op = theano.function(
      inputs=[thX, thY],
      outputs=[cost, prediction],
      updates=updates,
    )

    # for testing
    test_p_y_given_x = self.forward(thX, is_training=False)
    test_prediction = T.argmax(test_p_y_given_x, axis=1)

    self.predict = theano.function(
      inputs=[thX],
      outputs=test_prediction,
    )

    n_batches = N // batch_sz
    costs = []
    for i in range(epochs):
      if n_batches > 1:
        X, Y = shuffle(X, Y)
      for j in range(n_batches):
        Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
        Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

        c, p = train_op(Xbatch, Ybatch)
        costs.append(c)
        if (j+1) % print_period == 0:
          accuracy = np.mean(p == Ybatch)
          print("epoch:", i, "batch:", j, "n_batches:", n_batches, "cost:", c, "accuracy:", accuracy)

      print("Train acc:", self.score(X, Y), "Test acc:", self.score(Xtest, Ytest))
    
    if show_fig:
      plt.plot(costs)
      plt.show()

  def forward(self, X, is_training):
    out = X
    for h in self.layers[:-1]:
      out = h.forward(out, is_training)
    out = self.layers[-1].forward(out)
    return out

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(Y == P)



def main():
  # step 1: get the data and define all the usual variables
  Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

  ann = ANN([500, 300])
  ann.fit(Xtrain, Ytrain, Xtest, Ytest, show_fig=True)

  print("Train accuracy:", ann.score(Xtrain, Ytrain))
  print("Test accuracy:", ann.score(Xtest, Ytest))


if __name__ == '__main__':
  main()
