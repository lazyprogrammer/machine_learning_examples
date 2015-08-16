# A pedagogical example of k-means clustering.
# Creates a set of means (cluster centers) and
# generates random points around each center
# using spherical Gaussian noise. Then applies
# k-means clustering on the random points to
# allow us to compare the true clusters vs.
# predicted clusters.

# A link to the tutorial is here:
# http://lazyprogrammer.me/post/106693297569/k-means-clustering

import numpy as np
import matplotlib.pyplot as plt

class KMeans(object):
    def __init__(self, num_clusters, max_iter=1000, epsilon=0.0001):
        self.K = num_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon


    def fit(self, X):
        N = len(X)
        initial_indices = np.random.choice(np.arange(N), self.K)
        self.centers = np.array([X[i] for i in initial_indices])

        intermediate_predictions = []
        for i in xrange(self.max_iter):
            # there are only 2 steps for vanilla k-means clustering
            # 1. determing current cluster assignments
            # 2. find new cluster centers based on these cluster assignments

            # 1. determing current cluster assignments
            Y = self.predict(X)
            intermediate_predictions.append(Y)

            # 2. find new cluster centers
            self.centers, max_change = self.get_centers(X,Y)
            if max_change < self.epsilon:
                break
        return intermediate_predictions


    def predict(self, X):
        if len(X.shape) > 1:
            Y = np.zeros(len(X), dtype=int)
            for i,x in enumerate(X):
                min_c = -1
                min_dist = float("inf")
                for c,m in enumerate(self.centers):
                    d = np.linalg.norm(x - m)
                    if d < min_dist:
                        min_c = c
                        min_dist = d
                Y[i] = min_c
        else:
            Y = np.zeros(1, dtype=int)
            min_c = -1
            min_dist = float("inf")
            for c,m in enumerate(self.centers):
                d = np.linalg.norm(X - m)
                if d < min_dist:
                    min_c = c
                    min_dist = d
            Y[0] = min_c
        return Y


    # returns new centers calculated from labeled inputs
    # also returns 'max_change' between current centers
    # and new centers
    def get_centers(self, X, Y):
        X_partitioned = [[] for i in xrange(self.K)]

        for x,y in zip(X,Y):
            X_partitioned[y].append(x)

        new_centers = np.array([np.mean(x_part, 0) for x_part in X_partitioned])
        max_change = max(np.absolute(new_centers - self.centers).flatten())
        return new_centers, max_change


def test():
    # 5 cluster centers, 1 at origin, 4 at [+/-d, +/-d] from origin
    d = 3
    means = np.array([
        [0,0],
        [d,d],
        [-d,d],
        [-d,-d],
        [d,-d],
    ])
    cov = np.array([[1,0], [0,1]]) # circular

    print "actual centers:\n", means

    # generate random values
    samples_per_cluster = 50
    X = None
    color = 0
    colors = []
    for m in means:
        if X is not None:
            X = np.concatenate((X, np.random.multivariate_normal(m, cov, samples_per_cluster)))
        else:
            X = np.random.multivariate_normal(m, cov, samples_per_cluster)
        colors += [color]*samples_per_cluster
        color += 1

    plt.scatter(X[:,0], X[:,1], s=100, c=colors, alpha=0.5)
    plt.title("Target clusters")
    plt.show()

    print "colors.shape:", np.array(colors).shape
    print "X.shape:", X.shape

    plt.plot(colors)
    plt.title("Colors")
    plt.show()

    kmeans = KMeans(len(means))
    intermediate_predictions = kmeans.fit(X)

    print "predicted centers:\n", kmeans.centers

    c = kmeans.predict(np.array([[0.5, 0.5], [d, d]]))
    print "[0.5, 0.5] is predicted to be part of cluster %s" % c[0]
    print "[%d, %d] is predicted to be part of cluster %s" % (d, d, c[1])

    C = kmeans.predict(X)
    plt.scatter(X[:,0], X[:,1], c=C, s=100, alpha=0.5)
    plt.title("Predicted clusters")
    plt.show()

    for i, yhat in enumerate(intermediate_predictions):
        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=yhat, s=100, alpha=0.5)
        plt.title("Predictions at iteration %d" % (i+1))
        plt.show()


if __name__ == "__main__":
    test()
