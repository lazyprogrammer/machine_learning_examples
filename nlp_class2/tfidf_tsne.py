# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA
from datetime import datetime

import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

from util import find_analogies
from sklearn.feature_extraction.text import TfidfTransformer


def main():
    analogies_to_try = (
        ('king', 'man', 'woman'),
        ('france', 'paris', 'london'),
        ('france', 'paris', 'rome'),
        ('paris', 'france', 'italy'),
    )

    ### choose a data source ###
    # sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=1500)
    sentences, word2idx = get_wikipedia_data(n_files=3, n_vocab=2000, by_paragraph=True)
    # with open('tfidf_word2idx.json', 'w') as f:
    #     json.dump(word2idx, f)

    notfound = False
    for word_list in analogies_to_try:
        for w in word_list:
            if w not in word2idx:
                print("%s not found in vocab, remove it from \
                    analogies to try or increase vocab size" % w)
                notfound = True
    if notfound:
        exit()


    # build term document matrix
    V = len(word2idx)
    N = len(sentences)

    # create raw counts first
    A = np.zeros((V, N))
    print("V:", V, "N:", N)
    j = 0
    for sentence in sentences:
        for i in sentence:
            A[i,j] += 1
        j += 1
    print("finished getting raw counts")

    transformer = TfidfTransformer()
    A = transformer.fit_transform(A.T).T

    # tsne requires a dense array
    A = A.toarray()

    # map back to word in plot
    idx2word = {v:k for k, v in iteritems(word2idx)}

    # plot the data in 2-D
    tsne = TSNE()
    Z = tsne.fit_transform(A)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(V):
        try:
            plt.annotate(s=idx2word[i].encode("utf8").decode("utf8"), xy=(Z[i,0], Z[i,1]))
        except:
            print("bad string:", idx2word[i])
    plt.draw()

    ### multiple ways to create vectors for each word ###
    # 1) simply set it to the TF-IDF matrix
    # We = A

    # 2) create a higher-D word embedding
    tsne = TSNE(n_components=3)
    We = tsne.fit_transform(A)

    # 3) use a classic dimensionality reduction technique
    # svd = KernelPCA(n_components=20, kernel='rbf')
    # We = svd.fit_transform(A)

    for word_list in analogies_to_try:
        w1, w2, w3 = word_list
        find_analogies(w1, w2, w3, We, word2idx, idx2word)

    plt.show() # pause script until plot is closed


if __name__ == '__main__':
    main()
