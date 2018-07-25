# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


# simple installation with pip:
#
# just one line: pip install <....>.whl
#
# Linux
# https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python
#
# Windows
# https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python
#
# (No Mac)


# Note: useful to understand keras_example.py first

import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data, y2indicator


import cntk as C
from cntk.train import Trainer
from cntk.learners import adam
from cntk.ops import relu
from cntk.layers import Dense, Sequential
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *


# get the data, same as Theano + Tensorflow examples
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

# get shapes
N, D = Xtrain.shape
K = len(set(Ytrain))

# we want one-hot encoded labels
Ytrain = y2indicator(Ytrain)
Ytest = y2indicator(Ytest)

# split the data
X = X.astype(np.float32)
Y = Y.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Ytest = Ytest.astype(np.float32)


# the model will be a sequence of layers
model = Sequential(
  [
    Dense(500, activation=relu),
    Dense(300, activation=relu),
    Dense(K, activation=None),
  ]
)


# define the inputs and labels
inputs = C.input_variable(D, np.float32, name='inputs')
labels = C.input_variable(K, np.float32, name='labels')


# get the output
logits = model(inputs)


# define loss / metrics
# like Tensorflow the softmax is done
# internally (if needed), so all we need are the logits
ce = cross_entropy_with_softmax(logits, labels)
pe = classification_error(logits, labels)



# training config
batch_size = 32
epochs = 15
n_batches = len(Xtrain) // batch_size


# do the training

# specify the training algorithm
trainer = Trainer(logits, (ce, pe), adam(logits.parameters, lr=1e-2, momentum=0.9))


# helper function
def get_output(node, X, Y):
  ret = node.forward(dict(inputs=X, labels=Y))
  return list(ret[1].values())[0].mean()


costs = []
errors = []
test_costs = []
test_errors = []
for i in range(epochs):
  cost = 0
  err = 0
  for j in range(n_batches):
    Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
    Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

    ret = trainer.train_minibatch(
      dict(inputs=Xbatch, labels=Ybatch),
      outputs=(ce, pe)
    )

    cost += ret[1][ce].mean()
    err += ret[1][pe].mean()
  costs.append(cost / n_batches)
  errors.append(err / n_batches)

  # test set calculations
  test_cost = get_output(ce, Xtest, Ytest)
  test_err = get_output(pe, Xtest, Ytest)

  test_costs.append(test_cost)
  test_errors.append(test_err)

  print("epoch i:", i, "cost:", test_cost, "err:", test_err)

plt.plot(costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.title("cost")
plt.show()

plt.plot(errors, label='train error')
plt.plot(test_errors, label='test error')
plt.legend()
plt.title("error")
plt.show()
