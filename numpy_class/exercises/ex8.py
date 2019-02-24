# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

def get_spiral():
  # Idea: radius -> low...high
  #           (don't start at 0, otherwise points will be "mushed" at origin)
  #       angle = low...high proportional to radius
  #               [0, 2pi/6, 4pi/6, ..., 10pi/6] --> [pi/2, pi/3 + pi/2, ..., ]
  # x = rcos(theta), y = rsin(theta) as usual

  radius = np.linspace(1, 10, 100)
  thetas = np.empty((6, 100))
  for i in range(6):
      start_angle = np.pi*i / 3.0
      end_angle = start_angle + np.pi / 2
      points = np.linspace(start_angle, end_angle, 100)
      thetas[i] = points

  # convert into cartesian coordinates
  x1 = np.empty((6, 100))
  x2 = np.empty((6, 100))
  for i in range(6):
      x1[i] = radius * np.cos(thetas[i])
      x2[i] = radius * np.sin(thetas[i])

  # inputs
  X = np.empty((600, 2))
  X[:,0] = x1.flatten()
  X[:,1] = x2.flatten()

  # add noise
  X += np.random.randn(600, 2)*0.5

  # targets
  Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)
  return X, Y


X, Y = get_spiral()
plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()