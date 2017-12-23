# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.python.ops import rnn as rnn_module

######## This only works for pre-1.0 versions ##########
# from tensorflow.python.ops.rnn import rnn as get_rnn_output
# from tensorflow.python.ops.rnn_cell import BasicRNNCell, GRUCell
########################################################

########## This works for TensorFlow v1.0 ##############
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
########################################################

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs


def x2sequence(x, T, D, batch_sz):
  # Permuting batch_size and n_steps
  x = tf.transpose(x, (1, 0, 2))
  # Reshaping to (n_steps*batch_size, n_input)
  x = tf.reshape(x, (T*batch_sz, D))
  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
  # x = tf.split(0, T, x) # v0.1
  x = tf.split(x, T) # v1.0
  # print "type(x):", type(x)
  return x

class SimpleRNN:
  def __init__(self, M):
    self.M = M # hidden layer size


  def fit(self, X, Y, batch_sz=20, learning_rate=0.1, mu=0.9, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
    N, T, D = X.shape # X is of size N x T(n) x D
    K = len(set(Y.flatten()))
    M = self.M
    self.f = activation

    # initial weights
    # note: Wx, Wh, bh are all part of the RNN unit and will be created
    #       by BasicRNNCell
    Wo = init_weight(M, K).astype(np.float32)
    bo = np.zeros(K, dtype=np.float32)

    # make them tf variables
    self.Wo = tf.Variable(Wo)
    self.bo = tf.Variable(bo)

    # tf Graph input
    tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
    tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')

    # turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
    sequenceX = x2sequence(tfX, T, D, batch_sz)

    # create the simple rnn unit
    rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

    # Get rnn cell output
    # outputs, states = rnn_module.rnn(rnn_unit, sequenceX, dtype=tf.float32)
    outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

    # outputs are now of size (T, batch_sz, M)
    # so make it (batch_sz, T, M)
    outputs = tf.transpose(outputs, (1, 0, 2))
    outputs = tf.reshape(outputs, (T*batch_sz, M))

    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(outputs, self.Wo) + self.bo
    predict_op = tf.argmax(logits, 1)
    targets = tf.reshape(tfY, (T*batch_sz,))

    cost_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=targets
      )
    )
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)

    costs = []
    n_batches = N // batch_sz
    
    init = tf.global_variables_initializer()
    with tf.Session() as session:
      session.run(init)
      for i in range(epochs):
        X, Y = shuffle(X, Y)
        n_correct = 0
        cost = 0
        for j in range(n_batches):
          Xbatch = X[j*batch_sz:(j+1)*batch_sz]
          Ybatch = Y[j*batch_sz:(j+1)*batch_sz]
          
          _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
          cost += c
          for b in range(batch_sz):
            idx = (b + 1)*T - 1
            n_correct += (p[idx] == Ybatch[b][-1])
        if i % 10 == 0:
          print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
        if n_correct == N:
          print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
          break
        costs.append(cost)

    if show_fig:
      plt.plot(costs)
      plt.show()



def parity(B=12, learning_rate=1., epochs=1000):
  X, Y = all_parity_pairs_with_sequence_labels(B)

  rnn = SimpleRNN(4)
  rnn.fit(X, Y,
    batch_sz=len(Y),
    learning_rate=learning_rate,
    epochs=epochs,
    activation=tf.nn.sigmoid,
    show_fig=False
  )


if __name__ == '__main__':
  parity()

