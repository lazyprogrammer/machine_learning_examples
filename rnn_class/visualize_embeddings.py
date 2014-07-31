# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

def main(we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json', Model=PCA):
    We = np.load(we_file)
    V, D = We.shape
    with open(w2i_file) as f:
        word2idx = json.load(f)
    idx2word = {v:k for k,v in word2idx.iteritems()}

    model = Model()
    Z = model.fit_transform(We)
    plt.scatter(Z[:,0], Z[:,1])
    for i in xrange(V):
        plt.annotate(s=idx2word[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


if __name__ == '__main__':
    # main(Model=TSNE)

    # D=80, M=80
    # main(we_file='gru_nonorm_part1_word_embeddings.npy', w2i_file='gru_nonorm_part1_wikipedia_word2idx.json', Model=TSNE)
    main(we_file='working_files/batch_gru_word_embeddings.npy', w2i_file='working_files/batch_wikipedia_word2idx.json', Model=TSNE)
