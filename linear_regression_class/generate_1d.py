# generates 1-dimensional data for linear regression analysis
#
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python


import numpy as np

N = 100
with open('data_1d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=N)
    Y = 2*X + 1 + np.random.normal(scale=5, size=N)
    for i in xrange(N):
        f.write("%s,%s\n" % (X[i], Y[i]))

