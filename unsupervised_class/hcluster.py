# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage


def main():
    D = 2 # so we can visualize it more easily
    s = 4 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900 # number of samples
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    Z = linkage(X, 'ward')
    print "Z.shape:", Z.shape
    # Z has the format [idx1, idx2, dist, sample_count]
    # therefore, its size will be (N-1, 4)
    plt.title("Ward")
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'single')
    plt.title("Single")
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'complete')
    plt.title("Complete")
    dendrogram(Z)
    plt.show()


if __name__ == '__main__':
    main()
