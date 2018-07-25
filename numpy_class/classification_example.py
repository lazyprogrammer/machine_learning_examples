# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python
# YouTube direct link: http://bit.ly/2LENC50

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


# just in case we need it
import numpy as np


# import the function that will get the data
# yes, sklearn comes with built-in datasets!
from sklearn.datasets import load_breast_cancer

# load the data
data = load_breast_cancer()

# check the type of 'data'
type(data)

# note: it is a Bunch object
# this basically acts like a dictionary where you can treat the keys like attributes
data.keys()

# 'data' (the attribute) means the input data
data.data.shape
# it has 569 samples, 30 features

# 'targets'
data.target
# note how the targets are just 0s and 1s
# normally, when you have K targets, they are labeled 0..K-1

# their meaning is not lost
data.target_names

# there are also 569 corresponding targets
data.target.shape

# you can also determinw the meaning of each feature
data.feature_names


# normally we would put all of our imports at the top
# but this lets us tell a story
from sklearn.model_selection import train_test_split


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)


# instantiate a classifer and train it
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier()
model.fit(X_train, y_train)


# evaluate the model's performance
model.score(X_train, y_train)
model.score(X_test, y_test)


# how you can make predictions
predictions = model.predict(X_test)

# what did we get?
predictions

# manually check the accuracy of your predictions
N = len(y_test)
np.sum(predictions == y_test) / N # can also just call np.mean()



# we can even use deep learning to solve the same problem!
from sklearn.neural_network import MLPClassifier

# you'll learn why scaling is needed in a later course
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

model = MLPClassifier(max_iter=500)
model.fit(X_train2, y_train)


# evaluate the model's performance
model.score(X_train2, y_train)
model.score(X_test2, y_test)