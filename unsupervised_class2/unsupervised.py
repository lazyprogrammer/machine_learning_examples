# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from theano.tensor.shared_randomstreams import RandomStreams
from util import relu, error_rate, getKaggleMNIST, init_weights
from autoencoder import AutoEncoder
from rbm import RBM


class DBN(object):
    def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder):
        self.hidden_layers = []
        count = 0
        for M in hidden_layer_sizes:
            ae = UnsupervisedModel(M, count)
            self.hidden_layers.append(ae)
            count += 1

    def fit(self, X, pretrain_epochs=1):
        self.D = X.shape[1] # save for later

        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)

            # create current_input for the next layer
            current_input = ae.hidden_op(current_input)

        # return it here so we can use directly after fitting without calling forward
        return current_input

    def forward(self, X):
        Z = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(Z)
        return Z

    def fit_to_input(self, k, learning_rate=0.00001, mu=0.99, reg=10e-10, epochs=20000):
        # This is not very flexible, as you would ideally
        # like to be able to activate any node in any hidden
        # layer, not just the last layer.
        # Exercise for students: modify this function to be able
        # to activate neurons in the middle layers.
        X0 = init_weights((1, self.D))
        X = theano.shared(X0, 'X_shared')
        dX = theano.shared(np.zeros(X0.shape), 'dX_shared')
        Y = self.forward(X)
        # t = np.zeros(self.hidden_layers[-1].M)
        # t[k] = 1

        # # choose Y[0] b/c it's shape 1xD, we want just a D-size vector, not 1xD matrix
        # cost = -(t*T.log(Y[0]) + (1 - t)*(T.log(1 - Y[0]))).sum() + reg*(X * X).sum()

        cost = -T.log(Y[0,k]) + reg*(X * X).sum()

        updates = [
            (X, X + mu*dX - learning_rate*T.grad(cost, X)),
            (dX, mu*dX - learning_rate*T.grad(cost, X)),
        ]
        train = theano.function(
            inputs=[],
            outputs=[cost, Y],
            updates=updates,
        )

        costs = []
        bestX = None
        for i in xrange(epochs):
            if i % 1000 == 0:
                print "epoch:", i
            the_cost, out = train()
            if i == 0:
                print "out.shape:", out.shape
            costs.append(the_cost)
            # if the_cost < 10:
            #     break
            if the_cost > costs[-1] or np.isnan(the_cost):
                break

            bestX = X.get_value()
        print "len(costs):", len(costs), "max:", np.max(costs), "min:", np.min(costs)
        plt.plot(costs)
        plt.show()

        return bestX

    def save(self, filename):
        arrays = [p.get_value() for layer in self.hidden_layers for p in layer.params]
        np.savez(filename, *arrays)

    @staticmethod
    def load(filename, UnsupervisedModel=AutoEncoder):
        dbn = DBN([], UnsupervisedModel)
        npz = np.load(filename)
        dbn.hidden_layers = []
        count = 0
        for i in xrange(0, len(npz.files), 3):
            W = npz['arr_%s' % i]
            bh = npz['arr_%s' % (i + 1)]
            bo = npz['arr_%s' % (i + 2)]

            if i == 0:
                dbn.D = W.shape[0]

            ae = UnsupervisedModel.createFromArrays(W, bh, bo, count)
            dbn.hidden_layers.append(ae)
            count += 1
        return dbn


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dbn = DBN([1000, 750, 500], UnsupervisedModel=AutoEncoder)
    # dbn = DBN([1000, 750, 500, 10])
    output = dbn.fit(Xtrain, pretrain_epochs=2)
    print "output.shape", output.shape

    # sample before using t-SNE because it requires lots of RAM
    sample_size = 600
    tsne = TSNE()
    reduced = tsne.fit_transform(output[:sample_size])
    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain[:sample_size], alpha=0.5)
    plt.title("t-SNE visualization")
    plt.show()

    # t-SNE on raw data
    reduced = tsne.fit_transform(Xtrain[:sample_size])
    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain[:sample_size], alpha=0.5)
    plt.title("t-SNE visualization")
    plt.show()

    pca = PCA()
    reduced = pca.fit_transform(output)
    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain, alpha=0.5)
    plt.title("PCA visualization")
    plt.show()

if __name__ == '__main__':
    main()