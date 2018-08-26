from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from util import getKaggleMNIST, getKaggleFashionMNIST
from sklearn.neural_network import MLPClassifier


# get the data
Xtrain, Ytrain, Xtest, Ytest = getKaggleFashionMNIST()

# inspect your data
print(Xtrain.shape)
print(Ytrain.shape)

# look at an example
i = np.random.choice(Xtrain.shape[0])
plt.imshow(Xtrain[i].reshape(28, 28))
plt.title(Ytrain[i])
plt.show()

# instantiate the model
model = MLPClassifier()

# train the model
model.fit(Xtrain, Ytrain)

# evaluate the model
print(model.score(Xtrain, Ytrain))
print(model.score(Xtest, Ytest))

# for completion's sake, this is how you make predictions
Ptest = model.predict(Xtest)

# an alternate way to calculate accuracy
print(np.mean(Ptest == Ytest))

# get output probabilities
probs = model.predict_proba(Xtest)
print("np.argmax(probs, axis=1) == Ptest?", np.all(np.argmax(probs, axis=1) == Ptest))