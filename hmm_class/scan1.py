# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# theano scan example: calculate x^2

import numpy as np
import theano
import theano.tensor as T


x = T.vector('x')

def square(x):
  return x*x

outputs, updates = theano.scan(
  fn=square,
  sequences=x,
  n_steps=x.shape[0],
)

square_op = theano.function(
  inputs=[x],
  outputs=[outputs],
)

o_val = square_op(np.array([1, 2, 3, 4, 5]))

print "output:", o_val