'''
This is an example of a K-Nearest Neighbors classifier on MNIST data.

We try k=1...5 to show how we might choose the best k.

This is not production code!

See the tutorial here:
http://lazyprogrammer.me/post/114114892404/tutorial-k-nearest-neighbor-classifier-for-mnist
'''

import pandas as pd
import numpy as np
from sortedcontainers import SortedDict

Xtest = pd.read_csv("mnist_csv/Xtest.txt", header=None).as_matrix()
Xtrain = pd.read_csv("mnist_csv/Xtrain.txt", header=None).as_matrix()
Ytest = pd.read_csv("mnist_csv/label_test.txt", header=None).as_matrix().flatten()
Ytrain = pd.read_csv("mnist_csv/label_train.txt", header=None).as_matrix().flatten()

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points
            sd = SortedDict() # distance -> class
            for j,xt in enumerate(self.X): # training points
                d = np.linalg.norm(x - xt)
                # print d, sd
                if len(sd) < self.k:
                        sd[d] = self.y[j]
                else:
                    last = sd.viewkeys()[-1]
                    if d < last:
                        del sd[last]
                        sd[d] = self.y[j]
            # print "sd:", sd

            # vote
            votes = {}
            # print "viewvalues:", sd.viewvalues()
            for v in sd.viewvalues():
                # print "v:", v
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in votes.iteritems():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y


for k in (1,2,3,4,5):
    C = np.zeros((10,10), dtype=np.int)
    knn = KNN(k)
    knn.fit(Xtrain, Ytrain)
    Ypred = knn.predict(Xtest)
    for p,t in zip(Ypred, Ytest):
        C[t,p] += 1

    print "Confusion matrix for k = %d:" % k
    print C
    print "Accuracy:", np.trace(C) / 500.0
