# https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders
# https://www.udemy.com/deep-learning-gans-and-variational-autoencoders
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def softplus(x):
  # log1p(x) == log(1 + x)
  return np.log1p(np.exp(x))


# we're going to make a neural network
# with the layer sizes (4, 3, 2)
# like a toy version of a decoder

W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2*2)

# why 2 * 2?
# we need 2 components for the mean,
# and 2 components for the standard deviation!

# ignore bias terms for simplicity.

def forward(x, W1, W2):
  hidden = np.tanh(x.dot(W1))
  output = hidden.dot(W2) # no activation!
  mean = output[:2]
  stddev = softplus(output[2:])
  return mean, stddev


# make a random input
x = np.random.randn(4)

# get the parameters of the Gaussian
mean, stddev = forward(x, W1, W2)
print("mean:", mean)
print("stddev:", stddev)

# draw samples
samples = mvn.rvs(mean=mean, cov=stddev**2, size=10000)

# plot the samples
plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
plt.show()


