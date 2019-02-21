# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from scipy.io import loadmat
from sklearn.utils import shuffle

from benchmark import get_data, error_rate


# helper
# def y2indicator(Y):
#   N = len(Y)
#   K = len(set(Y))
#   I = np.zeros((N, K))
#   I[np.arange(N), Y] = 1
#   return I

def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3)
    # N = X.shape[-1]
    # out = np.zeros((N, 32, 32, 3), dtype=np.float32)
    # for i in xrange(N):
    #     for j in xrange(3):
    #         out[i, :, :, j] = X[:, :, j, i]
    # return out / 255
    return (X.transpose(3, 0, 1, 2) / 255.).astype(np.float32)


# get the data
train, test = get_data()

# Need to scale! don't leave as 0..255
# Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
# So flatten it and make it 0..9
# Also need indicator matrix for cost calculation
Xtrain = rearrange(train['X'])
Ytrain = train['y'].flatten() - 1
del train

Xtest  = rearrange(test['X'])
Ytest  = test['y'].flatten() - 1
del test



# get shapes
K = len(set(Ytrain))



# make the CNN
i = Input(shape=Xtrain.shape[1:])
x = Conv2D(filters=20, kernel_size=(5, 5))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=50, kernel_size=(5, 5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(units=500)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

model = Model(inputs=i, outputs=x)


# list of losses: https://keras.io/losses/
# list of optimizers: https://keras.io/optimizers/
# list of metrics: https://keras.io/metrics/
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# note: multiple ways to choose a backend
# either theano, tensorflow, or cntk
# https://keras.io/backend/


# gives us back a <keras.callbacks.History object at 0x112e61a90>
r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=32)
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


