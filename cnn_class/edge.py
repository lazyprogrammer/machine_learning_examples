# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the famous Lena image
img = mpimg.imread('lena.png')

# make it B&W
bw = img.mean(axis=2)

# Sobel operator - approximate gradient in X dir
Hx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

# Sobel operator - approximate gradient in Y dir
Hy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
], dtype=np.float32)

Gx = convolve2d(bw, Hx)
plt.imshow(Gx, cmap='gray')
plt.show()

Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap='gray')
plt.show()

# Gradient magnitude
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()

# The gradient's direction
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()