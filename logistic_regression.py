# The associated tutorial for this code is here:
# http://lazyprogrammer.me/post/117112987199/deep-learning-tutorial-part-1-3-logistic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Xtest = pd.read_csv("mnist_csv/Xtest.txt", header=None)
Xtest['ones'] = 1 # add bias term
Xtest = Xtest.as_matrix()
Xtrain = pd.read_csv("mnist_csv/Xtrain.txt", header=None)
Xtrain['ones'] = 1 # add bias term
Xtrain = Xtrain.as_matrix()
Ytest = pd.read_csv("mnist_csv/label_test.txt", header=None).as_matrix().flatten()
Ytrain = pd.read_csv("mnist_csv/label_train.txt", header=None).as_matrix().flatten()

class Logistic(object):
    def __init__(self, learningrate=0.00002, regularization=1, epochs=1000):
        self.learningrate = learningrate
        self.epochs = epochs
        self.regularization = regularization

    def fit(self, X, y):
        N, D = X.shape
        self.C = set(y)
        self.W = np.zeros((len(self.C),D))
        likelihoods = np.zeros(self.epochs)
        T = np.zeros((N, len(self.C))) # matrix of targets
        for n in xrange(N):
            T[n,y[n]] = 1
        for epoch in xrange(self.epochs):
            Y = self.forward(X)
            gradW = np.dot((T - Y).T, X) - self.regularization*self.W
            self.W += self.learningrate*gradW
            for n in xrange(N):
                likelihoods[epoch] += np.log(Y[n, y[n]])
            if epoch % 50 == 0:
                print epoch, likelihoods[epoch]
        return likelihoods

    def forward(self, X):
        N = len(X)
        Y = np.zeros((N, len(self.C)))
        # this assumes the class labels are 0,1,2,...|C|-1
        for n in xrange(N):
            z = np.exp(np.dot(self.W, X[n]))
            Y[n] = z / sum(z)
        return Y

    def predict(self, X):
        f = self.forward(X)
        return np.argmax(f, axis=1)

lr = Logistic(epochs=1000)
likelihoods = lr.fit(Xtrain, Ytrain)
plt.plot(likelihoods)
plt.show()

C = np.zeros((10,10), dtype=np.int)
Ypred = lr.predict(Xtest)
for p,t in zip(Ypred, Ytest):
    C[t,p] += 1

print "Confusion matrix:"
print C
print "Accuracy:", np.trace(C) / 500.0

