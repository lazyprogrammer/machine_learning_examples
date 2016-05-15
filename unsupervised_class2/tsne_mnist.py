# https://www.udemy.com/unsupervised-deep-learning-in-python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from util import getKaggleMNIST


def main():
    Xtrain, Ytrain, _, _ = getKaggleMNIST()

    sample_size = 1000
    X = Xtrain[:sample_size]
    Y = Ytrain[:sample_size]

    tsne = TSNE()
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()