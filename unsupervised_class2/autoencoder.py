# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weights


def T_shared_zeros_like32(p):
    # p is a Theano shared itself
    return theano.shared(np.zeros_like(p.get_value(), dtype=np.float32))

def momentum_updates(cost, params, mu, learning_rate):
    # momentum changes
    dparams = [T_shared_zeros_like32(p) for p in params]

    updates = []
    grads = T.grad(cost, params)
    for p, dp, g in zip(params, dparams, grads):
        dp_update = mu*dp - learning_rate*g
        p_update = p + dp_update

        updates.append((dp, dp_update))
        updates.append((p, p_update))
    return updates


class AutoEncoder(object):
    def __init__(self, M, an_id):
        self.M = M
        self.id = an_id

    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=1, batch_sz=100, show_fig=False):
        # cast to float
        mu = np.float32(mu)
        learning_rate = np.float32(learning_rate)

        N, D = X.shape
        n_batches = N // batch_sz

        W0 = init_weights((D, self.M))
        self.W = theano.shared(W0, 'W_%s' % self.id)
        self.bh = theano.shared(np.zeros(self.M, dtype=np.float32), 'bh_%s' % self.id)
        self.bo = theano.shared(np.zeros(D, dtype=np.float32), 'bo_%s' % self.id)
        self.params = [self.W, self.bh, self.bo]
        self.forward_params = [self.W, self.bh]

        # TODO: technically these should be reset before doing backprop
        self.dW = theano.shared(np.zeros(W0.shape), 'dW_%s' % self.id)
        self.dbh = theano.shared(np.zeros(self.M), 'dbh_%s' % self.id)
        self.dbo = theano.shared(np.zeros(D), 'dbo_%s' % self.id)
        self.dparams = [self.dW, self.dbh, self.dbo]
        self.forward_dparams = [self.dW, self.dbh]

        X_in = T.matrix('X_%s' % self.id)
        X_hat = self.forward_output(X_in)

        # attach it to the object so it can be used later
        # must be sigmoidal because the output is also a sigmoid
        H = T.nnet.sigmoid(X_in.dot(self.W) + self.bh)
        self.hidden_op = theano.function(
            inputs=[X_in],
            outputs=H,
        )

        # save this for later so we can call it to
        # create reconstructions of input
        self.predict = theano.function(
            inputs=[X_in],
            outputs=X_hat,
        )

        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).flatten().mean()
        cost_op = theano.function(
            inputs=[X_in],
            outputs=cost,
        )

        

        updates = momentum_updates(cost, self.params, mu, learning_rate)
        train_op = theano.function(
            inputs=[X_in],
            updates=updates,
        )

        costs = []
        print("training autoencoder: %s" % self.id)
        print("epochs to do:", epochs)
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                train_op(batch)
                the_cost = cost_op(batch) # technically we could also get the cost for Xtest here
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", the_cost)
                costs.append(the_cost)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_hidden(self, X):
        Z = T.nnet.sigmoid(X.dot(self.W) + self.bh)
        return Z

    def forward_output(self, X):
        Z = self.forward_hidden(X)
        Y = T.nnet.sigmoid(Z.dot(self.W.T) + self.bo)
        return Y

    @staticmethod
    def createFromArrays(W, bh, bo, an_id):
        ae = AutoEncoder(W.shape[1], an_id)
        ae.W = theano.shared(W, 'W_%s' % ae.id)
        ae.bh = theano.shared(bh, 'bh_%s' % ae.id)
        ae.bo = theano.shared(bo, 'bo_%s' % ae.id)
        ae.params = [ae.W, ae.bh, ae.bo]
        ae.forward_params = [ae.W, ae.bh]
        return ae


class DNN(object):
    def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder):
        self.hidden_layers = []
        count = 0
        for M in hidden_layer_sizes:
            ae = UnsupervisedModel(M, count)
            self.hidden_layers.append(ae)
            count += 1


    def fit(self, X, Y, Xtest, Ytest,
        pretrain=True,
        train_head_only=False,
        learning_rate=0.1,
        mu=0.99,
        reg=0.0,
        epochs=1,
        batch_sz=100):

        # cast to float32
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        reg = np.float32(reg)

        # greedy layer-wise training of autoencoders
        pretrain_epochs = 2
        if not pretrain:
            pretrain_epochs = 0

        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)

            # create current_input for the next layer
            current_input = ae.hidden_op(current_input)

        # initialize logistic regression layer
        N = len(Y)
        K = len(set(Y))
        W0 = init_weights((self.hidden_layers[-1].M, K))
        self.W = theano.shared(W0, "W_logreg")
        self.b = theano.shared(np.zeros(K, dtype=np.float32), "b_logreg")

        self.params = [self.W, self.b]
        if not train_head_only:
            for ae in self.hidden_layers:
                self.params += ae.forward_params

        X_in = T.matrix('X_in')
        targets = T.ivector('Targets')
        pY = self.forward(X_in)

        squared_magnitude = [(p*p).sum() for p in self.params]
        reg_cost = T.sum(squared_magnitude)
        cost = -T.mean( T.log(pY[T.arange(pY.shape[0]), targets]) ) + reg*reg_cost
        prediction = self.predict(X_in)
        cost_predict_op = theano.function(
            inputs=[X_in, targets],
            outputs=[cost, prediction],
        )

        updates = momentum_updates(cost, self.params, mu, learning_rate)
        train_op = theano.function(
            inputs=[X_in, targets],
            updates=updates,
        )

        n_batches = N // batch_sz
        costs = []
        print("supervised training...")
        for i in range(epochs):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                train_op(Xbatch, Ybatch)
                the_cost, the_prediction = cost_predict_op(Xtest, Ytest)
                error = error_rate(the_prediction, Ytest)
                print("j / n_batches:", j, "/", n_batches, "cost:", the_cost, "error:", error)
                costs.append(the_cost)
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return T.argmax(self.forward(X), axis=1)

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z

        # logistic layer
        Y = T.nnet.softmax(T.dot(current_input, self.W) + self.b)
        return Y


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    # dnn = DNN([1000, 750, 500])
    # dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3)
    # vs
    dnn = DNN([1000, 750, 500])
    dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=True, train_head_only=False, epochs=3)
    # note: try training the head only too! what does that mean?


def test_single_autoencoder():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

    autoencoder = AutoEncoder(300, 0)
    autoencoder.fit(Xtrain, epochs=2, show_fig=True)

    done = False
    while not done:
        i = np.random.choice(len(Xtest))
        x = Xtest[i]
        y = autoencoder.predict([x])
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28,28), cmap='gray')
        plt.title('Original')

        plt.subplot(1,2,2)
        plt.imshow(y.reshape(28,28), cmap='gray')
        plt.title('Reconstructed')

        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True


if __name__ == '__main__':
    main()
    # test_single_autoencoder()
