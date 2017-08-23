# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
# data from https://www.kaggle.com/benhamner/clinton-trump-tweets
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import pairwise_distances ### fast, but result not symmetric



# load stopwords
# selected after observing results without stopwords
stopwords = [
  'the',
  'about',
  'an',
  'and',
  'are',
  'at',
  'be',
  'can',
  'for',
  'from',
  'if',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'this',
  'to',
  'you',
  'your',
  'with',
]


# find urls and twitter usernames within a string
url_finder = re.compile(r"(?:\@|https?\://)\S+")


def filter_tweet(s):
  s = s.lower() # downcase
  s = url_finder.sub("", s) # remove urls and usernames
  return s



### load data ###
df = pd.read_csv('../large_files/tweets.csv')
text = df.text.tolist()
text = [filter_tweet(s) for s in text]


# transform the text into a data matrix
tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()


# subsample for efficiency
# remember: calculating distances is O(N^2)
N = X.shape[0]
idx = np.random.choice(N, size=2000, replace=False)
x = X[idx]
labels = df.handle[idx].tolist()


# proportions of each label
# so we can be confident that each is represented equally
pTrump = sum(1.0 if e == 'realDonaldTrump' else 0.0 for e in labels) / len(labels)
print("proportion @realDonaldTrump: %.3f" % pTrump)
print("proportion @HillaryClinton: %.3f" % (1 - pTrump))


# transform the data matrix into pairwise distances list
dist_array = pdist(x)


# calculate hierarchy
Z = linkage(dist_array, 'ward')
plt.title("Ward")
dendrogram(Z, labels=labels)
plt.show()

### hits max recursion depth
# Z = linkage(dist_array, 'single')
# plt.title("Single")
# dendrogram(Z, labels=labels)
# plt.show()

# Z = linkage(dist_array, 'complete')
# plt.title("Complete")
# dendrogram(Z, labels=labels)
# plt.show()


# convert labels to (1, 2), not (0, 1)
# since that's what's returned by fcluster
Y = np.array([1 if e == 'realDonaldTrump' else 2 for e in labels])


# get cluster assignments
# threshold 9 was chosen empirically to yield 2 clusters
C = fcluster(Z, 9, criterion='distance') # returns 1, 2, ..., K
categories = set(C)
# sanity check: should be {1, 2}
print("values in C:", categories)


### calculate the purity of our clusters ###
def purity(true_labels, cluster_assignments, categories):
  # maximum purity is 1, higher is better
  N = len(true_labels)

  total = 0.0
  for k in categories:
    max_intersection = 0
    for j in categories:
      intersection = ((cluster_assignments == k) & (true_labels == j)).sum()
      if intersection > max_intersection:
        max_intersection = intersection
    total += max_intersection
  return total / N

print("purity:", purity(Y, C, categories))


# we know the smaller cluster is the trump cluster
#
# important note: we call it the trump cluster from
# observing AFTER the fact that most tweets in this
# cluster are by trump
# we do NOT use those labels to create the cluster
#
# we create the clusters using a distance-based
# algorithm which knows nothing about the labels,
# just the tf-idf scores.
#
# it just so happens that most of the tweets in
# one of the clusters is by trump, and that this
# cluster is very small
if (C == 1).sum() < (C == 2).sum():
  d = 1
  h = 2
else:
  d = 2
  h = 1

actually_donald = ((C == d) & (Y == 1)).sum()
donald_cluster_size = (C == d).sum()
print("purity of @realDonaldTrump cluster:", float(actually_donald) / donald_cluster_size)

actually_hillary = ((C == h) & (Y == 2)).sum()
hillary_cluster_size = (C == h).sum()
print("purity of @HillaryClinton cluster:", float(actually_hillary) / hillary_cluster_size)


# just for interest, how would a classifier do?
# note: classification is always easier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, df.handle)
print("classifier score:", rf.score(X, df.handle))

# same as mnist
# classifier yields almost 100% accuracy
# but purity of clustering is much lower



# what words have the highest tf-idf in cluster 1? in cluster 2?
w2i = tfidf.vocabulary_

# tf-idf vectorizer todense() returns a matrix rather than array
# matrix always wants to be 2-D, so we convert to array in order to flatten
d_avg = np.array(x[C == d].mean(axis=0)).flatten()
d_sorted = sorted(w2i.keys(), key=lambda w: -d_avg[w2i[w]])

print("\nTop 10 'Donald cluster' words:")
print("\n".join(d_sorted[:10]))

h_avg = np.array(x[C == h].mean(axis=0)).flatten()
h_sorted = sorted(w2i.keys(), key=lambda w: -h_avg[w2i[w]])

print("\nTop 10 'Hillary cluster' words:")
print("\n".join(h_sorted[:10]))


