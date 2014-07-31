# https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders
# https://www.udemy.com/deep-learning-gans-and-variational-autoencoders
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import util
import numpy as np
import matplotlib.pyplot as plt

from vae_tf import VariationalAutoencoder
# from vae_theano import VariationalAutoencoder

if __name__ == '__main__':
  X, Y = util.get_mnist()
  # convert X to binary variable
  X = (X > 0.5).astype(np.float32)

  vae = VariationalAutoencoder(784, [200, 100, 2])
  vae.fit(X)

  Z = vae.transform(X)
  plt.scatter(Z[:,0], Z[:,1], c=Y)
  plt.show()


  # plot what image is reproduced for different parts of Z
  n = 20 # number of images per side
  x_values = np.linspace(-3, 3, n)
  y_values = np.linspace(-3, 3, n)
  image = np.empty((28 * n, 28 * n))

  # build Z first so we don't have to keep
  # re-calling the predict function
  # it is particularly slow in theano
  Z = []
  for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
      z = [x, y]
      Z.append(z)
  X_recon = vae.prior_predictive_with_input(Z)

  k = 0
  for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
      x_recon = X_recon[k]
      k += 1
      # convert from NxD == 1 x 784 --> 28 x 28
      x_recon = x_recon.reshape(28, 28)
      image[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_recon
  plt.imshow(image, cmap='gray')
  plt.show()
