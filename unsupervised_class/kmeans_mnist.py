# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python

# data is from https://www.kaggle.com/c/digit-recognizer
# each image is a D = 28x28 = 784 dimensional vector
# there are N = 42000 samples
# you can plot an image by reshaping to (28,28) and using plt.imshow()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import plot_k_means, get_simple_data
from datetime import datetime

def get_data(limit=None):
    print "Reading in and transforming data..."
    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def purity(Y, R):
    # maximum purity is 1, higher is better
    N, K = R.shape
    p = 0
    for k in xrange(K):
        best_target = -1 # we don't strictly need to store this
        max_intersection = 0
        for j in xrange(K):
            intersection = R[Y==j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
                best_target = j
        p += max_intersection
    return p / N


def DBI(X, M, R):
    # lower is better
    # N, D = X.shape
    # _, K = R.shape
    K, D = M.shape

    # get sigmas first
    sigma = np.zeros(K)
    for k in xrange(K):
        diffs = X - M[k] # should be NxD
        # assert(len(diffs.shape) == 2 and diffs.shape[1] == D)
        squared_distances = (diffs * diffs).sum(axis=1)
        # assert(len(squared_distances.shape) == 1 and len(squared_distances) != D)
        weighted_squared_distances = R[:,k]*squared_distances
        sigma[k] = np.sqrt(weighted_squared_distances).mean()

    # calculate Davies-Bouldin Index
    dbi = 0
    for k in xrange(K):
        max_ratio = 0
        for j in xrange(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K


def main():
    X, Y = get_data(1000)

    # simple data
    # X = get_simple_data()
    # Y = np.array([0]*300 + [1]*300 + [2]*300)

    print "Number of data points:", len(Y)
    # Note: I modified plot_k_means from the original
    # lecture to return means and responsibilities
    # print "performing k-means..."
    # t0 = datetime.now()
    M, R = plot_k_means(X, len(set(Y)))
    # print "k-means elapsed time:", (datetime.now() - t0)
    # Exercise: Try different values of K and compare the evaluation metrics
    print "Purity:", purity(Y, R)
    print "DBI:", DBI(X, M, R)


if __name__ == "__main__":
    main()
