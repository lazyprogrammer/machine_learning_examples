# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load in the data
df = pd.read_csv('../../large_files/train.csv')
data = df.values
X = data[:, 1:] # images
Y = data[:, 0] # labels

# loop through each label
for k in range(10):
  Xk = X[Y == k]

  # mean image
  Mk = Xk.mean(axis=0)

  # reshape into an image
  im = Mk.reshape(28, 28)

  # plot the image
  plt.imshow(im, cmap='gray')
  plt.title("Label: %s" % k)
  plt.show()
