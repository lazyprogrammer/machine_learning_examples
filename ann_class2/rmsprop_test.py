# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, y2indicator

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import theano
import theano.tensor as T


# RMSprop experiment
# to compare TF / Keras / Theano

N = 10
D = 2
X = np.random.randn(N, D).astype(np.float32)
w = np.array([0.5, -0.5], dtype=np.float32)
Y = X.dot(w) + 1
Y = Y.reshape(-1, 1)



# keras
# the model will be a sequence of layers
model = Sequential()
model.add(Dense(units=1, input_dim=D))


# copy the weights for later
weights = model.layers[0].get_weights()
w0 = weights[0].copy()
b0 = weights[1].copy()


model.compile(
  loss='mean_squared_error',
  optimizer='rmsprop',
)


r = model.fit(X, Y, epochs=15, batch_size=10)


# print the available keys
print(r.history.keys())



# tf
inputs = tf.placeholder(tf.float32, shape=(None, 2))
targets = tf.placeholder(tf.float32, shape=(None, 1))
tfw = tf.Variable(w0)
tfb = tf.Variable(b0)
pred = tf.matmul(inputs, tfw) + tfb

loss = tf.reduce_mean(tf.square(targets - pred))
train_op = tf.train.RMSPropOptimizer(1e-3, epsilon=1e-8).minimize(loss)

tflosses = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for e in range(15):
    _, l = sess.run([train_op, loss], feed_dict={inputs: X, targets: Y})
    tflosses.append(l)



# theano
def rmsprop(cost, params, lr=1e-3, decay=0.9, eps=1e-8):
  # return updates
  lr = np.float32(lr)
  decay = np.float32(decay)
  eps = np.float32(eps)

  updates = []
  grads = T.grad(cost, params)

  # tf-like
  # caches = [theano.shared(np.ones_like(p.get_value(), dtype=np.float32)) for p in params]

  # keras-like
  caches = [theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in params]
  
  new_caches = []
  for c, g in zip(caches, grads):
    new_c = decay*c + (np.float32(1) - decay)*g*g
    updates.append((c, new_c))
    new_caches.append(new_c)

  for p, new_c, g in zip(params, new_caches, grads):
    new_p = p - lr*g / T.sqrt(new_c + eps)
    updates.append((p, new_p))

  return updates

thX = T.matrix('X')
thY = T.matrix('Y')
thw = theano.shared(w0)
thb = theano.shared(b0)
thP = thX.dot(thw) + thb
cost = T.mean((thY - thP)**2)
params = [thw, thb]
updates = rmsprop(cost, params)

train_op = theano.function(
  inputs=[thX, thY],
  outputs=cost,
  updates=updates,
)

thlosses = []
for e in range(15):
  c = train_op(X, Y)
  thlosses.append(c)


# plot results
plt.plot(r.history['loss'], label='keras loss')
plt.plot(tflosses, label='tf loss')
plt.plot(thlosses, label='theano loss')
plt.legend()
plt.show()


