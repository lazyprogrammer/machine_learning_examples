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

# NOTE: do NOT name your file keras.py because it will conflict
# with importing keras

# installation is easy! just the usual "sudo pip(3) install keras"


# get the data, same as Theano + Tensorflow examples
# no need to split now, the fit() function will do it
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

# get shapes
N, D = Xtrain.shape
K = len(set(Ytrain))

# by default Keras wants one-hot encoded labels
# there's another cost function we can use
# where we can just pass in the integer labels directly
# just like Tensorflow / Theano
Ytrain = y2indicator(Ytrain)
Ytest = y2indicator(Ytest)


# the model will be a sequence of layers
model = Sequential()


# ANN with layers [784] -> [500] -> [300] -> [10]
model.add(Dense(units=500, input_dim=D))
model.add(Activation('relu'))
model.add(Dense(units=300)) # don't need to specify input_dim
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))


# list of losses: https://keras.io/losses/
# list of optimizers: https://keras.io/optimizers/
# list of metrics: https://keras.io/metrics/
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# note: multiple ways to choose a backend
# either theano, tensorflow, or cntk
# https://keras.io/backend/


# gives us back a <keras.callbacks.History object at 0x112e61a90>
r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=15, batch_size=32)
print("Returned:", r)

# print the available keys
# should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
print(r.history.keys())

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()


