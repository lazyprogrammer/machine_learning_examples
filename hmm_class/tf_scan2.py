# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# https://lazyprogrammer.me
# tensorflow scan example: calculate fibonacci

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf


# N = number of fibonacci numbers we want
# shape=() means scalar
N = tf.placeholder(tf.int32, shape=(), name='N')

# recurrence and loop
# notice how we don't use current_input at all!
def recurrence(last_output, current_input):
  return (last_output[1], last_output[0] + last_output[1])

fibonacci = tf.scan(
  fn=recurrence,
  elems=tf.range(N),
  initializer=(0, 1),
)

# run it!
with tf.Session() as session:
  o_val = session.run(fibonacci, feed_dict={N: 8})
  print("output:", o_val)
