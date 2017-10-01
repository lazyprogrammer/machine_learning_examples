# Theano basics.
# For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import theano.tensor as T

# just some different types of variables
c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')


# we can define a matrix multiplication
w = A.dot(v)

# how do these variables actually take on values?
import theano

matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

# let's import numpy so we can create real arrays
import numpy as np
A_val = np.array([[1,2], [3,4]])
v_val = np.array([5,6])

w_val = matrix_times_vector(A_val, v_val)
print(w_val)

# let's create a shared variable to we can do gradient descent
# this adds another layer of complexity to the theano function

x = theano.shared(20.0, 'x')

# the first argument is its initial value, the second is its name

# a cost function that has a minimum value
cost = x*x + x + 1

# in theano, you don't have to compute gradients yourself!
x_update = x - 0.3*T.grad(cost, x)

# x is not an "input", it's a thing you update
# in later examples, data and labels would go into the inputs
# and model params would go in the updates
# updates takes in a list of tuples, each tuple has 2 things in it:
# 1) the shared variable to update, 2) the update expression
train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

# write your own loop to call the training function.
# it has no arguments!
for i in range(25):
    cost_val = train()
    print(cost_val)

# print the optimal value of x
print(x.get_value())
