import numpy as np

N = 100
w = np.array([2, 3])
with open('data_2d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=(N,2))
    Y = np.dot(X, w) + 1 + np.random.normal(scale=5, size=N)
    for i in xrange(N):
        f.write("%s,%s,%s\n" % (X[i,0], X[i,1], Y[i]))
