# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main(we_file='glove_model_50.npz', w2i_file='glove_word2idx_50.json'):
    words = ['japan', 'japanese', 'england', 'english', 'australia', 'australian', 'china', 'chinese', 'italy', 'italian', 'french', 'france', 'spain', 'spanish']

    with open(w2i_file) as f:
        word2idx = json.load(f)

    npz = np.load(we_file)
    W = npz['arr_0']
    V = npz['arr_1']
    We = (W + V.T) / 2

    idx = [word2idx[w] for w in words]
    # We = We[idx]

    tsne = TSNE()
    Z = tsne.fit_transform(We)
    Z = Z[idx]
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(len(words)):
        plt.annotate(s=words[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


if __name__ == '__main__':
    main()
