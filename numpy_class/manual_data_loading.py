# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

# NOTE: in class, we assumed the current working directory
#       was linear_regression_class
#       in this file, we assume you are running the script
#       from the directory this file is in

import numpy as np

X = []

for line in open('../linear_regression_class/data_2d.csv'):
  row = line.split(',')
  sample = map(float, row)
  X.append(sample)

X = np.array(X)
print X