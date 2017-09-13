# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


if __name__ == '__main__':
  # define the centers of each Gaussian cloud
  centers = np.array([
    [ 1,  1,  1],
    [ 1,  1, -1],
    [ 1, -1,  1],
    [ 1, -1, -1],
    [-1,  1,  1],
    [-1,  1, -1],
    [-1, -1,  1],
    [-1, -1, -1],
  ])*3

  # create the clouds, Gaussian samples centered at
  # each of the centers we just made
  data = []
  pts_per_cloud = 100
  for c in centers:
    cloud = np.random.randn(pts_per_cloud, 3) + c
    data.append(cloud)
  data = np.concatenate(data)

  # visualize the clouds in 3-D
  # add colors / labels so we can track where the points go
  colors = np.array([[i]*pts_per_cloud for i in range(len(centers))]).flatten()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(data[:,0], data[:,1], data[:,2], c=colors)
  plt.show()


  # perform dimensionality reduction
  tsne = TSNE()
  transformed = tsne.fit_transform(data)

  # visualize the clouds in 2-D
  plt.scatter(transformed[:,0], transformed[:,1], c=colors)
  plt.show()



