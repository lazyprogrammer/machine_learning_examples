# https://lazyprogrammer.me
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense
from keras.optimizers import SGD, Adam


# make the original data
series1 = np.sin(0.1*np.arange(200))
series2 = np.sin(0.2*np.arange(200))

# plot it
plt.plot(series1)
plt.plot(series2)
plt.show()


### build the dataset
# let's see if we can use T past values to predict the next value
T = 10
D = 2
X = []
Y = []
for t in range(len(series1) - T - 1):
  x = [series1[t:t+T], series2[t:t+T]]
  # print("x[-1]:", x[-1])
  X.append(x)
  y = series1[t+T] + series2[t+T]
  # print("y:", y)
  Y.append(y)

X = np.array(X)
print("X.shape:", X.shape)
X = np.transpose(X, (0, 2, 1))
Y = np.array(Y)
N = len(X)



### many-to-one RNN

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
  X[:-N//2], Y[:-N//2],
  batch_size=32,
  epochs=80,
  validation_data=(X[-N//2:], Y[-N//2:]),
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# plot predictions vs targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title("many-to-one RNN")
plt.legend()
plt.show()

