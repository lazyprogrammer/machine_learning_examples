# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import random
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# our genetic code
code = ['A', 'T', 'C', 'G']


# convert list of integers to corresponding letters
def to_code(a):
  return [code[i] for i in a]


# distance between 2 DNA strands
def dist(a, b):
  return sum(i != j for i, j in zip(a, b))


# generate offspring by modifying some characters in the code
def generate_offspring(parent):
  return [maybe_modify(c) for c in parent]


# modify letter c with probability ~1/1000
def maybe_modify(c):
  if np.random.random() < 0.001:
    return np.random.choice(code)
  return c
  


# create 3 distinct ancestors
p1 = to_code(np.random.randint(4, size=1000))
p2 = to_code(np.random.randint(4, size=1000))
p3 = to_code(np.random.randint(4, size=1000))


# create offspring
num_generations = 99
max_offspring_per_generation = 1000
current_generation = [p1, p2, p3]
for i in range(num_generations):

  next_generation = []
  for parent in current_generation:
    # each parent will have between 1 and 3 children
    num_offspring = np.random.randint(3) + 1

    # generate the offspring
    for _ in range(num_offspring):
      child = generate_offspring(parent)
      next_generation.append(child)

  current_generation = next_generation

  # limit the number of offspring
  random.shuffle(current_generation)
  current_generation = current_generation[:max_offspring_per_generation]

  print("Finished creating generation %d / %d, size = %d" % (i + 2, num_generations + 1, len(current_generation)))



# create distance matrix
# note: you can also use scipy's pdist for this
# but NOT sklearn's pairwise_distances function
# which does not return a symmetric matrix
N = len(current_generation)
dist_matrix = np.zeros((N, N))
for i in range(N):
  for j in range(N):
    if i == j:
      continue
    elif j > i:
      a = current_generation[i]
      b = current_generation[j]
      dist_matrix[i,j] = dist(a, b)
    else:
      dist_matrix[i,j] = dist_matrix[j,i]

dist_array = ssd.squareform(dist_matrix)

Z = linkage(dist_array, 'ward')
plt.title("Ward")
dendrogram(Z)
plt.show()

Z = linkage(dist_array, 'single')
plt.title("Single")
dendrogram(Z)
plt.show()

Z = linkage(dist_array, 'complete')
plt.title("Complete")
dendrogram(Z)
plt.show()
