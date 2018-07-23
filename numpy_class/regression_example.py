# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python
# YouTube direct link: http://bit.ly/2LENC50

# Get the data from:
# https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


# just in case we need it
import numpy as np
import pandas as pd


# load the data
# important note: this is where we will usually put data files
df = pd.read_csv('../large_files/airfoil_self_noise.dat', sep='\t', header=None)

# check the data
df.head()
df.info()

# get the inputs
data = df[[0,1,2,3,4]].values

# get the outputs
target = df[5].values

# tiny update: pandas is moving from .as_matrix() to the equivalent .values


# normally we would put all of our imports at the top
# but this lets us tell a story
from sklearn.model_selection import train_test_split


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)


# instantiate a classifer and train it
from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(X_train, y_train)


# evaluate the model's performance
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


# how you can make predictions
predictions = model.predict(X_test)

# what did we get?
predictions



# we can even use random forest to solve the same problem!
from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)


# evaluate the model's performance
print(model2.score(X_train, y_train))
print(model2.score(X_test, y_test))




# we can even use deep learning to solve the same problem!
from sklearn.neural_network import MLPRegressor

# you'll learn why scaling is needed in a later course
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)
scaler2 = StandardScaler()
y_train2 = scaler2.fit_transform(np.expand_dims(y_train, -1)).ravel()
y_test2 = scaler2.fit_transform(np.expand_dims(y_test, -1)).ravel()

model = MLPRegressor(max_iter=500)
model.fit(X_train2, y_train2)


# evaluate the model's performance
print(model.score(X_train2, y_train2))
print(model.score(X_test2, y_test2))
# not as good as a random forest!
# but not as bad as linear regression
