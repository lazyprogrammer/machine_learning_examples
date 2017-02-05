# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow

import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
  return np.mean(p != t)


class TFLogistic:
  def __init__(self, savefile, D=None, K=None):
    self.savefile = savefile
    if D and K:
      # we can define some parts in the model to be able to make predictions
      self.build(D, K)

  def build(self, D, K):
    W0 = np.random.randn(D, K) * 2 / np.sqrt(D)
    b0 = np.zeros(K)

    # define variables and expressions
    self.inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
    self.targets = tf.placeholder(tf.int64, shape=(None,), name='targets')
    self.W = tf.Variable(W0.astype(np.float32), name='W')
    self.b = tf.Variable(b0.astype(np.float32), name='b')

    # variables must exist when calling this
    # try putting this line in the constructor and see what happens
    self.saver = tf.train.Saver({'W': self.W, 'b': self.b})

    logits = tf.matmul(self.inputs, self.W) + self.b
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.targets))
    self.predict_op = tf.argmax(logits, 1)
    return cost


  def fit(self, X, Y, Xtest, Ytest):
    N, D = X.shape
    K = len(set(Y))

    # hyperparams
    max_iter = 30
    lr = 10e-4
    mu = 0.9
    regularization = 10e-2
    batch_sz = 100
    n_batches = N / batch_sz

    cost = self.build(D, K)
    l2_penalty = regularization*tf.reduce_mean(self.W**2) / 2
    cost += l2_penalty
    train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)

    costs = []
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        for i in xrange(max_iter):
            for j in xrange(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={self.inputs: Xbatch, self.targets: Ybatch})
                if j % 100 == 0:
                    test_cost = session.run(cost, feed_dict={self.inputs: Xtest, self.targets: Ytest})
                    Ptest = session.run(self.predict_op, feed_dict={self.inputs: Xtest})
                    err = error_rate(Ptest, Ytest)
                    print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err)
                    costs.append(test_cost)

        # save the model
        self.saver.save(session, self.savefile)

    # save dimensions for later
    self.D = D
    self.K = K

    plt.plot(costs)
    plt.show()


  def predict(self, X):
    with tf.Session() as session:
      # restore the model
      self.saver.restore(session, self.savefile)
      P = session.run(self.predict_op, feed_dict={self.inputs: X})
    return P


  def score(self, X, Y):
    return 1 - error_rate(self.predict(X), Y)

  def save(self, filename):
    j = {
      'D': self.D,
      'K': self.K,
      'model': self.savefile
    }
    with open(filename, 'w') as f:
      json.dump(j, f)

  @staticmethod
  def load(filename):
    with open(filename) as f:
      j = json.load(f)
    return TFLogistic(j['model'], j['D'], j['K'])


def main():
    X, Y = get_normalized_data()

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:,]
    Ytest  = Y[-1000:]

    model = TFLogistic("tf.model")
    model.fit(Xtrain, Ytrain, Xtest, Ytest)

    # test out restoring the model via the predict function
    print "final train accuracy:", model.score(Xtrain, Ytrain)
    print "final test accuracy:", model.score(Xtest, Ytest)

    # save the model
    model.save("my_trained_model.json")

    # load and score again
    model = TFLogistic.load("my_trained_model.json")
    print "final train accuracy (after reload):", model.score(Xtrain, Ytrain)
    print "final test accuracy (after reload):", model.score(Xtest, Ytest)


if __name__ == '__main__':
    main()