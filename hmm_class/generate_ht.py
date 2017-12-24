# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Generate discrete data from an HMM.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np


symbol_map = ['H', 'T']
pi = np.array([0.5, 0.5])
A = np.array([[0.1, 0.9], [0.8, 0.2]])
B = np.array([[0.6, 0.4], [0.3, 0.7]])
M, V = B.shape


def generate_sequence(N):
    s = np.random.choice(xrange(M), p=pi) # initial state
    x = np.random.choice(xrange(V), p=B[s]) # initial observation
    sequence = [x]
    for n in range(N-1):
        s = np.random.choice(xrange(M), p=A[s]) # next state
        x = np.random.choice(xrange(V), p=B[s]) # next observation
        sequence.append(x)
    return sequence


def main():
    with open('coin_data.txt', 'w') as f:
        for n in range(50):
            sequence = generate_sequence(30)
            sequence = ''.join(symbol_map[s] for s in sequence)
            print sequence
            f.write("%s\n" % sequence)


if __name__ == '__main__':
    main()