# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances



def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


# slow version
# def find_analogies(w1, w2, w3, We, word2idx):
#     king = We[word2idx[w1]]
#     man = We[word2idx[w2]]
#     woman = We[word2idx[w3]]
#     v0 = king - man + woman

#     def dist1(a, b):
#         return np.linalg.norm(a - b)
#     def dist2(a, b):
#         return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

#     for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
#         min_dist = float('inf')
#         best_word = ''
#         for word, idx in iteritems(word2idx):
#             if word not in (w1, w2, w3):
#                 v1 = We[idx]
#                 d = dist(v0, v1)
#                 if d < min_dist:
#                     min_dist = d
#                     best_word = word
#         print("closest match by", name, "distance:", best_word)
#         print(w1, "-", w2, "=", best_word, "-", w3)

# fast version
def find_analogies(w1, w2, w3, We, word2idx, idx2word):
    V, D = We.shape

    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    for dist in ('euclidean', 'cosine'):
        distances = pairwise_distances(v0.reshape(1, D), We, metric=dist).reshape(V)
        # idx = distances.argmin()
        # best_word = idx2word[idx]
        idx = distances.argsort()[:4]
        best_idx = -1
        keep_out = [word2idx[w] for w in (w1, w2, w3)]
        for i in idx:
            if i not in keep_out:
                best_idx = i
                break
        best_word = idx2word[best_idx]


        print("closest match by", dist, "distance:", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)


class Tree:
    def __init__(self, word, label):
        self.left = None
        self.right = None
        self.word = word
        self.label = label


def display_tree(t, lvl=0):
    prefix = ''.join(['>']*lvl)
    if t.word is not None:
        print("%s%s %s" % (prefix, t.label, t.word))
    else:
        print("%s%s -" % (prefix, t.label))
        # if t.left is None or t.right is None:
        #     raise Exception("Tree node has no word but left and right child are None")
    if t.left:
        display_tree(t.left, lvl + 1)
    if t.right:
        display_tree(t.right, lvl + 1)


current_idx = 0
def str2tree(s, word2idx):
    # take a string that starts with ( and MAYBE ends with )
    # return the tree that it represents
    # EXAMPLE: "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
    # NOTE: not every node has 2 children (possibly not correct ??)
    # NOTE: not every node has a word
    # NOTE: every node has a label
    # NOTE: labels are 0,1,2,3,4
    # NOTE: only leaf nodes have words
    # s[0] = (, s[1] = label, s[2] = space, s[3] = character or (

    # print "Input string:", s, "len:", len(s)

    global current_idx

    label = int(s[1])
    if s[3] == '(':
        t = Tree(None, label)
        # try:

        # find the string that represents left child
        # it can include trailing characters we don't need, because we'll only look up to )
        child_s = s[3:]
        t.left = str2tree(child_s, word2idx)

        # find the string that represents right child
        # can contain multiple ((( )))
        # left child is completely represented when we've closed as many as we've opened
        # we stop at 1 because the first opening paren represents the current node, not children nodes
        i = 0
        depth = 0
        for c in s:
            i += 1
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 1:
                    break
        # print "index of right child", i

        t.right = str2tree(s[i+1:], word2idx)

        # except Exception as e:
        #     print "Exception:", e
        #     print "Input string:", s
        #     raise e

        # if t.left is None or t.right is None:
        #     raise Exception("Tree node has no word but left and right child are None")
        return t
    else:
        # this has a word, so it's a leaf
        r = s.split(')', 1)[0]
        word = r[3:].lower()
        # print "word found:", word

        if word not in word2idx:
            word2idx[word] = current_idx
            current_idx += 1

        t = Tree(word2idx[word], label)
        return t


def get_ptb_data():
    # like the wikipedia dataset, I want to return 2 things:
    # word2idx mapping, sentences
    # here the sentences should be Tree objects

    if not os.path.exists('../large_files/trees'):
        print("Please create ../large_files/trees relative to this file.")
        print("train.txt and test.txt should be stored in there.")
        print("Please download the data from http://nlp.stanford.edu/sentiment/")
        exit()
    elif not os.path.exists('../large_files/trees/train.txt'):
        print("train.txt is not in ../large_files/trees/train.txt")
        print("Please download the data from http://nlp.stanford.edu/sentiment/")
        exit()
    elif not os.path.exists('../large_files/trees/test.txt'):
        print("test.txt is not in ../large_files/trees/test.txt")
        print("Please download the data from http://nlp.stanford.edu/sentiment/")
        exit()

    word2idx = {}
    train = []
    test = []

    # train set first
    for line in open('../large_files/trees/train.txt'):
        line = line.rstrip()
        if line:
            t = str2tree(line, word2idx)
            # if t.word is None and t.left is None and t.right is None:
            #     print "sentence:", line
            # display_tree(t)
            # print ""
            train.append(t)
            # break

    # test set
    for line in open('../large_files/trees/test.txt'):
        line = line.rstrip()
        if line:
            t = str2tree(line, word2idx)
            test.append(t)
    return train, test, word2idx

# get_ptb_data()
