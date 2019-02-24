# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

A = np.array([
  [0.3, 0.6, 0.1],
  [0.5, 0.2, 0.3],
  [0.4, 0.1, 0.5]])

v = np.ones(3) / 3

num_iters = 25
distances = np.zeros(num_iters)
for i in range(num_iters):
  v2 = v.dot(A)
  d = np.linalg.norm(v2 - v)
  distances[i] = d
  v = v2

plt.plot(distances)
plt.show()