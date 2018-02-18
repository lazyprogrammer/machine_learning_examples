# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ConvBlock:
  def __init__(self):
    pass

  def predict(self, X):
    pass


if __name__ == '__main__':
  conv_block = ConvBlock()


  # make a fake image
  X = np.random.random((1, 224, 224, 3))

  init = tf.global_variables_initializer()
  with tf.Session() as session:
    conv_block.session = session
    session.run(init)

    output = conv_block.predict(X):
    print("output.shape:", output.shape)