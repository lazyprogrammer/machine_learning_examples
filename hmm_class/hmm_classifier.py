# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Demonstrate how HMMs can be used for classification.
import string
import numpy as np
import matplotlib.pyplot as plt

from hmmd_theano import HMM
from sklearn.utils import shuffle
from nltk import pos_tag, word_tokenize

class HMMClassifier:
    def __init__(self):
        pass

    def fit(self, X, Y, V):
        K = len(set(Y)) # number of classes - assume 0..K-1
        self.models = []
        self.priors = []
        for k in xrange(K):
            # gather all the training data for this class
            thisX = [x for x, y in zip(X, Y) if y == k]
            C = len(thisX)
            self.priors.append(np.log(C))

            hmm = HMM(5)
            hmm.fit(thisX, V=V, p_cost=0.1, print_period=1, learning_rate=1e-4, max_iter=100)
            self.models.append(hmm)

    def score(self, X, Y):
        N = len(Y)
        correct = 0
        for x, y in zip(X, Y):
            lls = [hmm.log_likelihood(x) + prior for hmm, prior in zip(self.models, self.priors)]
            p = np.argmax(lls)
            if p == y:
                correct += 1
        return float(correct) / N


# def remove_punctuation(s):
#     return s.translate(None, string.punctuation)

def get_tags(s):
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples]

def get_data():
    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('robert_frost.txt', 'edgar_allan_poe.txt'), (0, 1)):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line:
                print line
                # tokens = remove_punctuation(line.lower()).split()
                tokens = get_tags(line)
                if len(tokens) > 1:
                    # scan doesn't work nice here, technically could fix...
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    print count
                    if count >= 50:
                        break
    print "Vocabulary:", word2idx.keys()
    return X, Y, current_idx
        

def main():
    X, Y, V = get_data()
    # print "Finished loading data"
    print "len(X):", len(X)
    print "Vocabulary size:", V
    X, Y = shuffle(X, Y)
    N = 20 # number to test
    Xtrain, Ytrain = X[:-N], Y[:-N]
    Xtest, Ytest = X[-N:], Y[-N:]

    model = HMMClassifier()
    model.fit(Xtrain, Ytrain, V)
    print "Score:", model.score(Xtest, Ytest)


if __name__ == '__main__':
    main()
