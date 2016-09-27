import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# we need to skip the 3 footer rows
# skipfooter does not work with the default engine, 'c'
# so we need to explicitly set it to 'python'
df = pd.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)

# rename the columns because they are ridiculous
df.columns = ['month', 'num_passengers']

# plot the data so we know what it looks like
plt.plot(df.num_passengers)
plt.show()

# let's try with only the time series itself
series = df.num_passengers.as_matrix()

# def myr2(T, Y, Ym):
#     sse = (T - Y).dot(T - Y)
#     sst = (T - Ym).dot(T - Ym)
#     return 1 - sse / sst

# let's see if we can use D past values to predict the next value
N = len(series)
for D in (2,3,4,5,6,7):
    n = N - D
    X = np.zeros((n, D))
    for d in xrange(D):
        X[:,d] = series[d:d+n]
    Y = series[D:D+n]

    # print "X.shape:", X.shape
    # print "Y.shape:", Y.shape

    print "series length:", n
    Xtrain = X[:n/2]
    Ytrain = Y[:n/2]
    Xtest = X[n/2:]
    Ytest = Y[n/2:]

    # print "Xtrain.shape:", Xtrain.shape
    # print "Ytrain.shape:", Ytrain.shape

    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    print "train score:", model.score(Xtrain, Ytrain)

    # Ytrain_mean = Ytrain.mean()
    # print "myr2 train:", myr2(Ytrain, model.predict(Xtrain), Ytrain_mean)

    # print "Xtest.shape:", Xtest.shape
    # print "Ytest.shape:", Ytest.shape
    print "test score:", model.score(Xtest, Ytest)

    # Ytest_mean = Ytest.mean()
    # print "myr2 test w/ Ytrain mean:", myr2(Ytest, model.predict(Xtest), Ytrain_mean)
    # print "myr2 test w/ Ytest mean:", myr2(Ytest, model.predict(Xtest), Ytest_mean) # this is the one score uses

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
