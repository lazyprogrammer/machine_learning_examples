import numpy as np
from numpy import linalg


NUM_SAMPLES = 1000


class LinearRegression(object):
    def fit(self, X, y):
        D = len(X[0]) + 1
        P = np.zeros((D, D))
        Q = np.zeros(D)
        for xi, yi in zip(X, y):
            x = np.concatenate([[1], xi])
            P += np.outer(x, x)
            Q += x*yi
        self.beta = np.dot(linalg.inv(P), Q)


    def score(self, X, y):
        meany = np.mean(y)
        SStot = 0
        SSres = 0
        for xi, yi in zip(X, y):
            x = np.concatenate([[1], xi])
            res = yi - np.dot(self.beta, x)
            tot = yi - meany
            SSres += res*res
            SStot += tot*tot
        return 1 - SSres/SStot


    def predict(self, x):
        # prepend 1, dot product with beta
        return np.inner(self.beta, np.concatenate([[1], x]))


def main():
    # create a bunch of random data for X-axis
    # uniformly generate numbers in [-50, 50]
    X = 100*np.random.random([NUM_SAMPLES, 2]) - 50

    # create a bunch of random data for Y-axis
    # let's say y = 5x1 - 2x2 + 3 + noise
    # true beta is then: [3, 5, -2]
    Y = np.fromiter((5*x1 - 2*x2 + 3 for x1, x2 in X), np.float, count=NUM_SAMPLES)
    Y += np.random.standard_normal(NUM_SAMPLES)

    # fit
    lr = LinearRegression()
    lr.fit(X,Y)
    print "beta estimated: %s" % lr.beta

    r2 = lr.score(X,Y)
    print "R-square is: %s" % r2

    # predict
    x = (100, 100)
    h = lr.predict(x)
    y = 5*x[0] - 2*x[1] + 3
    print "Extrapolated prediction: %.2f\nActual: %.2f" % (h, y)


if __name__ == "__main__":
    main()
