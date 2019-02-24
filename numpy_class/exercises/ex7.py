# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

def get_donut():
  N = 2000
  R_inner = 5
  R_outer = 10

  # distance from origin is radius + random normal
  # angle theta is uniformly distributed between (0, 2pi)
  R1 = np.random.randn(N//2) + R_inner
  theta = 2*np.pi*np.random.random(N//2)
  X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

  R2 = np.random.randn(N//2) + R_outer
  theta = 2*np.pi*np.random.random(N//2)
  X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

  X = np.concatenate([ X_inner, X_outer ])
  Y = np.array([0]*(N//2) + [1]*(N//2))
  return X, Y

X, Y = get_donut()
plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()