# https://deeplearningcourses.com/c/support-vector-machines-in-python
# https://www.udemy.com/support-vector-machines-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def getKaggleMNIST():
  # MNIST data:
  # column 0 is labels
  # column 1-785 is data, with values 0 .. 255
  # total size of CSV: (42000, 784)
  train = pd.read_csv('../large_files/train.csv').values.astype(np.float32)
  train = shuffle(train)

  Xtrain = train[:-1000,1:]
  Ytrain = train[:-1000,0].astype(np.int32)

  Xtest  = train[-1000:,1:]
  Ytest  = train[-1000:,0].astype(np.int32)

  # scale the data
  Xtrain /= 255.
  Xtest /= 255.
  # scaler = StandardScaler()
  # Xtrain = scaler.fit_transform(Xtrain)
  # Xtest  = scaler.transform(Xtest)

  return Xtrain, Ytrain, Xtest, Ytest


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


def get_xor():
  X = np.zeros((200, 2))
  X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
  X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
  X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
  X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
  Y = np.array([0]*100 + [1]*100)
  return X, Y


def get_donut():
  N = 200
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


def get_clouds():
  N = 1000
  c1 = np.array([2, 2])
  c2 = np.array([-2, -2])
  # c1 = np.array([0, 3])
  # c2 = np.array([0, 0])
  X1 = np.random.randn(N, 2) + c1
  X2 = np.random.randn(N, 2) + c2
  X = np.vstack((X1, X2))
  Y = np.array([-1]*N + [1]*N)
  return X, Y


def plot_decision_boundary(model, resolution=100, colors=('b', 'k', 'r')):
  np.warnings.filterwarnings('ignore')
  fig, ax = plt.subplots()

  # Generate coordinate grid of shape [resolution x resolution]
  # and evaluate the model over the entire space
  x_range = np.linspace(model.Xtrain[:,0].min(), model.Xtrain[:,0].max(), resolution)
  y_range = np.linspace(model.Xtrain[:,1].min(), model.Xtrain[:,1].max(), resolution)
  grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
  grid = np.array(grid).reshape(len(x_range), len(y_range))
  
  # Plot decision contours using grid and
  # make a scatter plot of training data
  ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
             linestyles=('--', '-', '--'), colors=colors)
  ax.scatter(model.Xtrain[:,0], model.Xtrain[:,1],
             c=model.Ytrain, lw=0, alpha=0.3, cmap='seismic')
  
  # Plot support vectors (non-zero alphas)
  # as circled points (linewidth > 0)
  mask = model.alphas > 0.
  ax.scatter(model.Xtrain[:,0][mask], model.Xtrain[:,1][mask],
             c=model.Ytrain[mask], cmap='seismic')

  # debug
  ax.scatter([0], [0], c='black', marker='x')
  
  plt.show()
