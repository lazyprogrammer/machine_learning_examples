# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# https://lazyprogrammer.me
# tensorflow scan example - low pass filter

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# original sequence is a noisy sine wave
original = np.sin(np.linspace(0, 3*np.pi, 300))
X = 2*np.random.randn(300) + original
plt.plot(X)
plt.title("original")
plt.show()

# set up placeholders
decay = tf.placeholder(tf.float32, shape=(), name='decay')
sequence = tf.placeholder(tf.float32, shape=(None,), name='sequence')

# the recurrence function and loop
def recurrence(last, x):
  return (1.0-decay)*x + decay*last

lpf = tf.scan(
  fn=recurrence,
  elems=sequence,
  initializer=0.0, # sequence[0] to use the first value of the sequence
)

# run it!
with tf.Session() as session:
  Y = session.run(lpf, feed_dict={sequence: X, decay: 0.97})

  plt.plot(Y)
  plt.plot(original)
  plt.title("filtered")
  plt.show()
