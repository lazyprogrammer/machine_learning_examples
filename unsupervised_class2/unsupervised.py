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
        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)

            # create current_input for the next layer
            current_input = ae.hidden_op(current_input)

        # return it here so we can use directly after fitting without calling forward
        return current_input

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z
        return current_input

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