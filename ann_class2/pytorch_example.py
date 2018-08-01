# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

# Note: is helpful to look at keras_example.py first


import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data

import torch
from torch.autograd import Variable
from torch import optim



# get the data, same as Theano + Tensorflow examples
# no need to split now, the fit() function will do it
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

# get shapes
_, D = Xtrain.shape
K = len(set(Ytrain))

# Note: no need to convert Y to indicator matrix


# the model will be a sequence of layers
model = torch.nn.Sequential()


# ANN with layers [784] -> [500] -> [300] -> [10]
model.add_module("dense1", torch.nn.Linear(D, 500))
model.add_module("relu1", torch.nn.ReLU())
model.add_module("dense2", torch.nn.Linear(500, 300))
model.add_module("relu2", torch.nn.ReLU())
model.add_module("dense3", torch.nn.Linear(300, K))
# Note: no final softmax!
# just like Tensorflow, it's included in cross-entropy function



# define a loss function
# other loss functions can be found here:
# http://pytorch.org/docs/master/nn.html#loss-functions
loss = torch.nn.CrossEntropyLoss(size_average=True)



# define an optimizer
# other optimizers can be found here:
# http://pytorch.org/docs/master/optim.html
optimizer = optim.Adam(model.parameters())



# define the training procedure
# i.e. one step of gradient descent
# there are lots of steps
# so we encapsulate it in a function
# Note: inputs and labels are torch tensors
def train(model, loss, optimizer, inputs, labels):
  # https://discuss.pytorch.org/t/why-is-it-recommended-to-wrap-your-data-with-variable-each-step-of-the-iterations-rather-than-before-training-starts/12683
  inputs = Variable(inputs, requires_grad=False)
  labels = Variable(labels, requires_grad=False)

  # Reset gradient
  # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/7
  optimizer.zero_grad()

  # Forward
  logits = model.forward(inputs)
  output = loss.forward(logits, labels)

  # Backward
  output.backward()

  # Update parameters
  optimizer.step()

  # what's the difference between backward() and step()?
  # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
  return output.item()


# define the prediction procedure
# also encapsulate these steps
# Note: inputs is a torch tensor
def predict(model, inputs):
  inputs = Variable(inputs, requires_grad=False)
  logits = model.forward(inputs)
  return logits.data.numpy().argmax(axis=1)



### prepare for training loop ###

# convert the data arrays into torch tensors
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()



epochs = 15
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size

costs = []
test_accuracies = []
for i in range(epochs):
  cost = 0.
  for j in range(n_batches):
    Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
    Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]
    cost += train(model, loss, optimizer, Xbatch, Ybatch)

  Ypred = predict(model, Xtest)
  acc = np.mean(Ytest == Ypred)
  print("Epoch: %d, cost: %f, acc: %.2f" % (i, cost / n_batches, acc))

  # for plotting
  costs.append(cost / n_batches)
  test_accuracies.append(acc)


# EXERCISE: plot test cost + training accuracy too

# plot the results
plt.plot(costs)
plt.title('Training cost')
plt.show()

plt.plot(test_accuracies)
plt.title('Test accuracies')
plt.show()
