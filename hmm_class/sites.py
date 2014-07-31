# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Create a Markov model for site data.
import numpy as np

transitions = {}
row_sums = {}

# collect counts
for line in open('site_data.csv'):
    s, e = line.rstrip().split(',')
    transitions[(s, e)] = transitions.get((s, e), 0.) + 1
    row_sums[s] = row_sums.get(s, 0.) + 1

# normalize
for k, v in transitions.iteritems():
    s, e = k
    transitions[k] = v / row_sums[s]

# initial state distribution
print "initial state distribution:"
for k, v in transitions.iteritems():
    s, e = k
    if s == '-1':
        print e, v

# which page has the highest bounce?
for k, v in transitions.iteritems():
    s, e = k
    if e == 'B':
        print "bounce rate for %s: %s" % (s, v)
