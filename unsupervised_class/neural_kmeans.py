import numpy as np
import matplotlib.pyplot as plt
from kmeans import get_simple_data
from sklearn.preprocessing import StandardScaler


# get the data and standardize it
X = get_simple_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# get shapes
N, D = X.shape
K = 3

# initialize parameters
W = np.random.randn(D, K)

# set hyperparameters
n_epochs = 100
learning_rate = 0.001
losses = []

# training loop
for i in range(n_epochs):
  loss = 0
  for j in range(N):
    h = W.T.dot(X[j]) # K-length vector
    k = np.argmax(h) # winning neuron

    # accumulate loss
    loss += (W[:,k] - X[j]).dot(W[:,k] - X[j])

    # weight update
    W[:,k] += learning_rate * (X[j] - W[:,k])

  losses.append(loss)


# plot losses
plt.plot(losses)
plt.show()

# show cluster assignments
H = np.argmax(X.dot(W), axis=1)
plt.scatter(X[:,0], X[:,1], c=H, alpha=0.5)
plt.show()
