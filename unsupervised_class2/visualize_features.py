import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from util import relu, error_rate, getKaggleMNIST, init_weights
from unsupervised import DBN
from rbm import RBM


def main(loadfile=None, savefile=None):
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    if loadfile:
        dbn = DBN([1000, 750, 500, 10]) # AutoEncoder is default
        dbn = DBN([1000, 750, 500, 10], UnsupervisedModel=RBM)
        dbn.fit(Xtrain, pretrain_epochs=15)
    else:
        dbn.load(loadfile)

    if savefile:
        dbn.save(savefile)

    # initial weight is D x M
    # W = dbn.hidden_layers[0].W.eval()
    # for i in xrange(dbn.hidden_layers[0].M):
    #     imgplot = plt.imshow(W[:,i].reshape(28, 28), cmap='gray')
    #     plt.show()
    #     should_quit = raw_input("Show more? Enter 'n' to quit\n")
    #     if should_quit == 'n':
    #         break

    # TODO: save the weights so I can initialize from them later
    #       and just do the last step

    # print features learned in the last layer
    for k in xrange(dbn.hidden_layers[-1].M):
        # activate the kth node
        X = dbn.fit_to_input(k)
        imgplot = plt.imshow(X.reshape(28, 28), cmap='gray')
        plt.show()
        should_quit = raw_input("Show more? Enter 'n' to quit\n")
        if should_quit == 'n':
            break


if __name__ == '__main__':
    main()