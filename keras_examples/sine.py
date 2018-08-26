# https://lazyprogrammer.me
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense
import keras.backend as K

from keras.optimizers import SGD, Adam


# make the original data
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1

# plot it
plt.plot(series)
plt.show()


### build the dataset
# let's see if we can use T past values to predict the next value
T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T - 1):
  x = series[t:t+T]
  # print("x[-1]:", x[-1])
  X.append(x)
  y = series[t+T]
  # print("y:", y)
  Y.append(y)

X = np.array(X)
Y = np.array(Y)
N = len(X)



### many-to-one RNN
inputs = np.expand_dims(X, -1)

# make the RNN
i = Input(shape=(T, D))
x = SimpleRNN(5)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
  inputs[:-N//2], Y[:-N//2],
  batch_size=32,
  epochs=80,
  validation_data=(inputs[-N//2:], Y[-N//2:]),
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# plot predictions vs targets
outputs = model.predict(inputs)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title("many-to-one RNN")
plt.legend()
plt.show()

