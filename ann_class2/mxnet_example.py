# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


# installation is just one line:
# https://mxnet.incubator.apache.org/get_started/install.html
#
# Mac:
# pip install mxnet
#
# Linux (GPU):
# pip install mxnet-cu80
#
# Windows (a little more involved):
# https://mxnet.incubator.apache.org/get_started/windows_setup.html

import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data, y2indicator



# get the data, same as Theano + Tensorflow examples
# no need to split now, the fit() function will do it
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

# get shapes
N, D = Xtrain.shape
K = len(set(Ytrain))

# training config
batch_size = 32
epochs = 15


# convert the data into a format appropriate for input into mxnet
train_iterator = mx.io.NDArrayIter(
  Xtrain,
  Ytrain,
  batch_size,
  shuffle=True
)
test_iterator = mx.io.NDArrayIter(Xtest, Ytest, batch_size)



# define a placeholder to represent the inputs
data = mx.sym.var('data')


# define the model architecture
a1 = mx.sym.FullyConnected(data=data, num_hidden=500)
z1 = mx.sym.Activation(data=a1, act_type="relu")
a2 = mx.sym.FullyConnected(data=z1, num_hidden = 300)
z2 = mx.sym.Activation(data=a2, act_type="relu")
a3 = mx.sym.FullyConnected(data=z2, num_hidden=K)
y  = mx.sym.SoftmaxOutput(data=a3, name='softmax')




# train it

# required in order for progress to be printed
import logging
logging.getLogger().setLevel(logging.DEBUG)

# use mx.gpu() if you have gpu
model = mx.mod.Module(symbol=y, context=mx.cpu())
model.fit(
  train_iterator, # train data
  eval_data=test_iterator,  # validation data
  optimizer=mx.optimizer.Adam(),
  eval_metric='acc',  # report accuracy during training
  batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
  num_epoch=epochs,
)
# no return value
# list of optimizers: https://mxnet.incubator.apache.org/api/python/optimization.html


# test it
# predict accuracy of mlp
acc = mx.metric.Accuracy()
model.score(test_iterator, acc)
print(acc)
print(acc.get())


# currently, there is no good way to plot the training loss / accuracy history
# https://github.com/apache/incubator-mxnet/issues/2873
#
# some have suggested parsing the logs
# https://github.com/apache/incubator-mxnet/blob/master/example/kaggle-ndsb1/training_curves.py
