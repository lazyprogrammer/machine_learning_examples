# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt


# generate unlabeled data
N = 2000
X = np.random.random((N, 2))*2 - 1

# generate labels
Y = np.zeros(N)
Y[(X[:,0] < 0) & (X[:,1] > 0)] = 1
Y[(X[:,0] > 0) & (X[:,1] < 0)] = 1

# plot it
plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()