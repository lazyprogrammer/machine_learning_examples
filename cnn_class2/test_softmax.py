# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import tensorflow as tf
import numpy as np
import keras
import keras.backend as K

def custom_softmax(x):
  m = tf.reduce_max(x, 1)
  x = x - m
  e = tf.exp(x)
  return e / tf.reduce_sum(e, -1)


a = np.random.randn(1, 1000)

tfy = tf.nn.softmax(a)
ky = keras.activations.softmax(K.variable(a))
tfc = custom_softmax(a)

session = K.get_session()

tfy_ = session.run(tfy)
ky_ = session.run(ky)
tfc_ = session.run(tfc)

print("tf vs k", np.abs(tfy_ - ky_).sum())
print("tf vs custom", np.abs(tfy_ - tfc_).sum())
print("custom vs k", np.abs(tfc_ - ky_).sum())