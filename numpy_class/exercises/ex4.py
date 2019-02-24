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

# shuffle the images
np.random.shuffle(data)

X = data[:, 1:] # images
Y = data[:, 0] # labels


# define rotate functions
def rotate1(im):
  return np.rot90(im, 3)

def rotate2(im):
  H, W = im.shape
  im2 = np.zeros((W, H))
  for i in range(H):
    for j in range(W):
      im2[j,H - i - 1] = im[i,j]
  return im2


for i in range(X.shape[0]):
  # get the image
  im = X[i].reshape(28, 28)

  # flip the image
  # im = rotate1(im)
  im = rotate2(im)

  # plot the image
  plt.imshow(im, cmap='gray')
  plt.title("Label: %s" % Y[i])
  plt.show()

  ans = input("Continue? [Y/n]: ")
  if ans and ans[0].lower() == 'n':
    break
