# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from util import relu, error_rate, getKaggleMNIST, init_weights
from autoencoder import DNN


class RBM(object):
    def __init__(self, M, an_id):
        self.M = M
        self.id = an_id
        self.rng = RandomStreams()

    def fit(self, X, learning_rate=0.1, epochs=1, batch_sz=100, show_fig=False):
        N, D = X.shape
        n_batches = N / batch_sz

        W0 = init_weights((D, self.M))
        self.W = theano.shared(W0, 'W_%s' % self.id)
        self.c = theano.shared(np.zeros(self.M), 'c_%s' % self.id)
        self.b = theano.shared(np.zeros(D), 'b_%s' % self.id)
        self.params = [self.W, self.c, self.b]
        self.forward_params = [self.W, self.c]

        # we won't use this to fit the RBM but we will use these for backpropagation later
        # TODO: technically they should be reset before doing backprop
        self.dW = theano.shared(np.zeros(W0.shape), 'dW_%s' % self.id)
        self.dc = theano.shared(np.zeros(self.M), 'dbh_%s' % self.id)
        self.db = theano.shared(np.zeros(D), 'dbo_%s' % self.id)
        self.dparams = [self.dW, self.dc, self.db]
        self.forward_dparams = [self.dW, self.dc]

        X_in = T.matrix('X_%s' % self.id)

        # attach it to the object so it can be used later
        # must be sigmoidal because the output is also a sigmoid
        H = T.nnet.sigmoid(X_in.dot(self.W) + self.c)
        self.hidden_op = theano.function(
            inputs=[X_in],
            outputs=H,
        )

        # we won't use this cost to do any updates
        # but we would like to see how this cost function changes
        # as we do contrastive divergence
        X_hat = self.forward_output(X_in)
        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).sum() / (batch_sz * D)
        cost_op = theano.function(
            inputs=[X_in],
            outputs=cost,
        )

        # do one round of Gibbs sampling to obtain X_sample
        H = self.sample_h_given_v(X_in)
        X_sample = self.sample_v_given_h(H)

        # define the objective, updates, and train function
        objective = T.mean(self.free_energy(X_in)) - T.mean(self.free_energy(X_sample))

        # need to consider X_sample constant because you can't take the gradient of random numbers in Theano
        updates = [(p, p - learning_rate*T.grad(objective, p, consider_constant=[X_sample])) for p in self.params]
        train_op = theano.function(
            inputs=[X_in],
            updates=updates,
        )

        costs = []
        print "training rbm: %s" % self.id
        for i in xrange(epochs):
            print "epoch:", i
            X = shuffle(X)
            for j in xrange(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                train_op(batch)
                the_cost = cost_op(X)  # technically we could also get the cost for Xtest here
                print "j / n_batches:", j, "/", n_batches, "cost:", the_cost
                costs.append(the_cost)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def free_energy(self, V):
        return  -V.dot(self.b) - T.sum(T.log(1 + T.exp(V.dot(self.W) + self.c)), axis=1)

    def sample_h_given_v(self, V):
        p_h_given_v = T.nnet.sigmoid(V.dot(self.W) + self.c)
        h_sample = self.rng.binomial(size=p_h_given_v.shape, n=1, p=p_h_given_v)
        return h_sample

    def sample_v_given_h(self, H):
        p_v_given_h = T.nnet.sigmoid(H.dot(self.W.T) + self.b)
        v_sample = self.rng.binomial(size=p_v_given_h.shape, n=1, p=p_v_given_h)
        return v_sample

    def forward_hidden(self, X):
        return T.nnet.sigmoid(X.dot(self.W) + self.c)

    def forward_output(self, X):
        Z = self.forward_hidden(X)
        Y = T.nnet.sigmoid(Z.dot(self.W.T) + self.b)
        return Y

    @staticmethod
    def createFromArrays(W, c, b, an_id):
        rbm = AutoEncoder(W.shape[1], an_id)
        rbm.W = theano.shared(W, 'W_%s' % rbm.id)
        rbm.c = theano.shared(c, 'c_%s' % rbm.id)
        rbm.b = theano.shared(b, 'b_%s' % rbm.id)
        rbm.params = [rbm.W, rbm.c, rbm.b]
        rbm.forward_params = [rbm.W, rbm.c]
        return rbm


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dnn = DNN([1000, 750, 500], UnsupervisedModel=RBM)
    dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3)

    # we compare with no pretraining in autoencoder.py


if __name__ == '__main__':
    main()