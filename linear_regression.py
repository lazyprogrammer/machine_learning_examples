import numpy as np
from numpy import linalg
from regressor import Regressor

NUM_SAMPLES = 1000


class LinearRegression(Regressor):
    def fit(self, X, y):
        N = len(X)
        X = np.concatenate((np.array([[1]*N]).T, X), axis=1)
        self.beta = np.dot( np.dot( np.linalg.inv(np.dot(X.T,X)), X.T ), y )


    def predict(self, X):
        # prepend 1, dot product with beta
        N = len(X)
        X = np.concatenate((np.array([[1]*N]).T, X), axis=1)
        return np.inner(self.beta, X)


def test():
    # create a bunch of random data for X-axis
    # uniformly generate 2-D vectors in [-50, 50]
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
    h = lr.predict(np.array([x]))
    y = 5*x[0] - 2*x[1] + 3
    print "Extrapolated prediction: %.2f\nActual: %.2f" % (h, y)


if __name__ == "__main__":
    test()
