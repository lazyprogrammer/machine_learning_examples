# The corresponding tutorial for this code was released EXCLUSIVELY as a bonus
# If you want to learn about future bonuses, please sign up for my newsletter at:
# https://lazyprogrammer.me

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from sklearn.utils import shuffle

def init_weight(M1, M2):
    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)

def myr2(T, Y):
    Ym = T.mean()
    sse = (T - Y).dot(T - Y)
    sst = (T - Ym).dot(T - Ym)
    return 1 - sse / sst

class HiddenLayer(object):
    def __init__(self, M1, M2, f, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return self.f(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, activation=T.tanh, learning_rate=10e-4, mu=0.5, reg=0, epochs=5000, batch_sz=None, print_period=100, show_fig=True):
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        # initialize hidden layers
        N, D = X.shape
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W = np.random.randn(M1) / np.sqrt(M1)
        b = 0.0
        self.W = theano.shared(W, 'W_last')
        self.b = theano.shared(b, 'b_last')

        if batch_sz is None:
            batch_sz = N

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.vector('Y')
        Yhat = self.forward(thX)

        rcost = reg*T.mean([(p*p).sum() for p in self.params])
        cost = T.mean((thY - Yhat).dot(thY - Yhat)) + rcost
        prediction = self.forward(thX)
        grads = T.grad(cost, self.params)

        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
        )

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
        )

        n_batches = N / batch_sz
        # print "N:", N, "batch_sz:", batch_sz
        # print "n_batches:", n_batches
        costs = []
        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            for j in xrange(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)
                if (j+1) % print_period == 0:
                    print "i:", i, "j:", j, "nb:", n_batches, "cost:", c
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return Z.dot(self.W) + self.b

    def score(self, X, Y):
        Yhat = self.predict_op(X)
        return myr2(Y, Yhat)

    def predict(self, X):
        return self.predict_op(X)

# we need to skip the 3 footer rows
# skipfooter does not work with the default engine, 'c'
# so we need to explicitly set it to 'python'
df = pd.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)

# rename the columns because they are ridiculous
df.columns = ['month', 'num_passengers']

# plot the data so we know what it looks like
# plt.plot(df.num_passengers)
# plt.show()

# let's try with only the time series itself
series = df.num_passengers.as_matrix()
# series = (series - series.mean()) / series.std() # normalize the values so they have mean 0 and variance 1
series = series.astype(np.float32)
series = series - series.min()
series = series / series.max()

# let's see if we can use D past values to predict the next value
N = len(series)
for D in (2,3,4,5,6,7):
    n = N - D
    X = np.zeros((n, D))
    for d in xrange(D):
        X[:,d] = series[d:d+n]
    Y = series[D:D+n]

    print "series length:", n
    Xtrain = X[:n/2]
    Ytrain = Y[:n/2]
    Xtest = X[n/2:]
    Ytest = Y[n/2:]

    model = ANN([200])
    model.fit(Xtrain, Ytrain, activation=T.tanh)
    print "train score:", model.score(Xtrain, Ytrain)
    print "test score:", model.score(Xtest, Ytest)

    # plot the prediction with true values
    plt.plot(series)

    train_series = np.empty(n)
    train_series[:n/2] = model.predict(Xtrain) 
    train_series[n/2:] = np.nan
    # prepend d nan's since the train series is only of size N - D
    plt.plot(np.concatenate([np.full(d, np.nan), train_series]))

    test_series = np.empty(n)
    test_series[:n/2] = np.nan
    test_series[n/2:] = model.predict(Xtest)
    plt.plot(np.concatenate([np.full(d, np.nan), test_series]))

    plt.show()
