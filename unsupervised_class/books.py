# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import networkx as nx
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE
from sklearn.feature_extraction.text import TfidfTransformer

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('../nlp_class/all_book_titles.txt')]

# copy tokenizer from sentiment example
stopwords = set(w.rstrip() for w in open('../nlp_class/stopwords.txt'))
# add more stopwords specific to this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })
def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
print("num titles:", len(titles))
print("first title:", titles[0])
for title in titles:
    try:
        title = title.encode('ascii', 'ignore') # this will throw exception if bad characters
        title = title.decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)



# now let's create our input matrices - just indicator variables for this example - works better than proportions
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    return x

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N)) # terms will go along rows, documents along columns
i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1

def d(u, v):
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        # method 1
        # for n in range(len(X)):
        #     cost += R[n,k]*d(M[k], X[n])

        # method 2
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost

def plot_k_means(X, K, index_word_map, max_iter=20, beta=1.0, show_plots=True):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))
    exponents = np.empty((N, K))

    # initialize M to random
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(max_iter)
    for i in range(max_iter):
        # step 1: determine assignments / resposibilities
        # is this inefficient?
        for k in range(K):
            for n in range(N):
                # R[n,k] = np.exp(-beta*d(M[k], X[n])) / np.sum( np.exp(-beta*d(M[j], X[n])) for j in range(K) )
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))

        R = exponents / exponents.sum(axis=1, keepdims=True)

        # step 2: recalculate means
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()

        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 10e-5:
                break

    if show_plots:
        # plt.plot(costs)
        # plt.title("Costs")
        # plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.figure(figsize=(80.0, 80.0))
        plt.scatter(X[:,0], X[:,1], s=300, alpha=0.9, c=colors)
        annotate1(X, index_word_map)
        # plt.show()
        plt.savefig("test.png")


    # print out the clusters
    hard_responsibilities = np.argmax(R, axis=1) # is an N-size array of cluster identities
    # let's "reverse" the order so it's cluster identity -> word index
    cluster2word = {}
    for i in range(len(hard_responsibilities)):
      word = index_word_map[i]
      cluster = hard_responsibilities[i]
      if cluster not in cluster2word:
        cluster2word[cluster] = []
      cluster2word[cluster].append(word)

    # print out the words grouped by cluster
    for cluster, wordlist in cluster2word.items():
      print("cluster", cluster, "->", wordlist)

    return M, R

# def annotate2(X, index_word_map, k=0.1):
#   N, D = X.shape

#   # create graph
#   G = nx.DiGraph()
#   data_nodes = []
#   init_pos = {}
#   for i in range(N):
#     x, y = X[i]
#     label = index_word_map[i]
#     data_str = 'data_{0}'.format(label)
#     G.add_node(data_str)
#     G.add_node(label)
#     G.add_edge(label, data_str)
#     data_nodes.append(data_str)
#     init_pos[data_str] = (x, y)
#     init_pos[label] = (x, y)
#   pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

#   # undo spring_layout's rescaling
#   pos_after = np.vstack([pos[d] for d in data_nodes])
#   pos_before = np.vstack([init_pos[d] for d in data_nodes])
#   scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
#   scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
#   shift = np.array([shift_x, shift_y])
#   for key, val in pos.items():
#     pos[key] = (val*scale) + shift


#   for label, data_str in G.edges():
#     plt.annotate(
#       label,
#       xy=pos[data_str], xycoords='data',
#       xytext=pos[label], textcoords='data',
#       arrowprops=dict(arrowstyle="->", color='black'),
#     )

#   # expand limits
#   all_pos = np.vstack(pos.values())
#   x_span, y_span = np.ptp(all_pos, axis=0)
#   mins = np.min(all_pos-x_span*0.15, 0)
#   maxs = np.max(all_pos+y_span*0.15, 0)
#   plt.xlim([mins[0], maxs[0]])
#   plt.ylim([mins[1], maxs[1]])


def annotate1(X, index_word_map, eps=0.1):
  N, D = X.shape
  placed = np.empty((N, D))
  for i in range(N):
    x, y = X[i]

    # if x, y is too close to something already plotted, move it
    close = []

    x, y = X[i]
    for retry in range(3):
      for j in range(i):
        diff = np.array([x, y]) - placed[j]

        # if something is close, append it to the close list
        if diff.dot(diff) < eps:
          close.append(placed[j])

      if close:
        # then the close list is not empty
        x += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
        y += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
        close = [] # so we can start again with an empty list
      else:
        # nothing close, let's break
        break

    placed[i] = (x, y)

    plt.annotate(
      s=index_word_map[i],
      xy=(X[i,0], X[i,1]),
      xytext=(x, y),
      arrowprops={
        'arrowstyle' : '->',
        'color' : 'black',
      }
    )

print("vocab size:", current_index)

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

reducer = TSNE()
Z = reducer.fit_transform(X)
plot_k_means(Z[:,:2], current_index//10, index_word_map, show_plots=True)
