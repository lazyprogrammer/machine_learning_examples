# demonstrates how to calculate the output of a logistic unit using numpy.
# the data X and weight matrix w are randomly generated from a
# standard normal distribution.
#
# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-logistic-regression-in-python
# https://www.udemy.com/data-science-logistic-regression-in-python

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np

N = 100
D = 2


X = np.random.randn(N,D)
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print(sigmoid(z))
