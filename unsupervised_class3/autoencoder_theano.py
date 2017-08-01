# https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders
# https://www.udemy.com/deep-learning-gans-and-variational-autoencoders
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import util
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


class Autoencoder:
  def __init__(self, D, M):
    # represents a batch of training data
    self.X = T.matrix('X')

    # input -> hidden
    self.W = theano.shared(np.random.randn(D, M) * 2 / np.sqrt(M))
    self.b = theano.shared(np.zeros(M))

    # hidden -> output
    self.V = theano.shared(np.random.randn(M, D) * 2 / np.sqrt(D))
    self.c = theano.shared(np.zeros(D))

    # construct the reconstruction
    self.Z = T.nnet.relu(self.X.dot(self.W) + self.b)
    self.X_hat = T.nnet.sigmoid(self.Z.dot(self.V) + self.c)

    # compute the cost
    self.cost = T.sum(
      T.nnet.binary_crossentropy(
        output=self.X_hat,
        target=self.X,
      )
    )

    # define the updates
    params = [self.W, self.b, self.V, self.c]
    grads = T.grad(self.cost, params)

    # rmsprop
    decay = 0.9
    learning_rate = 0.001

    # for rmsprop
    cache = [theano.shared(np.ones_like(p.get_value())) for p in params]
    new_cache = [decay*c + (1-decay)*g*g for p, c, g in zip(params, cache, grads)]

    updates = [
        (c, new_c) for c, new_c in zip(cache, new_cache)
    ] + [
        (p, p - learning_rate*g/T.sqrt(new_c + 1e-10)) for p, new_c, g in zip(params, new_cache, grads)
    ]


    # now define callable functions
    self.train_op = theano.function(
      inputs=[self.X],
      outputs=self.cost,
      updates=updates
    )

    self.predict = theano.function(
      inputs=[self.X],
      outputs=self.X_hat
    )

  def fit(self, X, epochs=30, batch_sz=64):
    costs = []
    n_batches = len(X) // batch_sz
    print("n_batches:", n_batches)
    for i in range(epochs):
      print("epoch:", i)
      np.random.shuffle(X)
      for j in range(n_batches):
        batch = X[j*batch_sz:(j+1)*batch_sz]
        c = self.train_op(batch)
        c /= batch_sz # just debugging
        costs.append(c)
        if j % 100 == 0:
          print("iter: %d, cost: %.3f" % (j, c))
    plt.plot(costs)
    plt.show()


def main():
  X, Y = util.get_mnist()

  model = Autoencoder(784, 300)
  model.fit(X)

  # plot reconstruction
  done = False
  while not done:
    i = np.random.choice(len(X))
    x = X[i]
    im = model.predict([x]).reshape(28, 28)
    plt.subplot(1,2,1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(im, cmap='gray')
    plt.title("Reconstruction")
    plt.show()

    ans = input("Generate another?")
    if ans and ans[0] in ('n' or 'N'):
      done = True

if __name__ == '__main__':
  main()

