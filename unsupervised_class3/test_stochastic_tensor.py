# https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders
# https://www.udemy.com/deep-learning-gans-and-variational-autoencoders

# a simple script to see what StochasticTensor outputs
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal

# sample N samples from N(5,3*3)
N = 10000
mean = np.ones(N)*5
scale = np.ones(N)*3


I = tf.Variable(np.ones(N))


with st.value_type(st.SampleValue()):
  X = st.StochasticTensor(Normal(loc=mean, scale=scale))

# cannot session.run a stochastic tensor
# but we can session.run a tensor
Y = I * X


init_op = tf.global_variables_initializer()
with tf.Session() as session:
  session.run(init_op)
  Y_val = session.run(Y)

  print("Sample mean:", Y_val.mean())
  print("Sample std dev:", Y_val.std())

  plt.hist(Y_val, bins=20)
  plt.show()
