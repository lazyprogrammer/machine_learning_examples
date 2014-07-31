# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python

# data from https://github.com/aritter/twitter_nlp/blob/master/data/annotated/ner.txt
# data2 from http://schwa.org/projects/resources/wiki/Wikiner#WikiGold

import numpy as np
from sklearn.utils import shuffle
from pos_baseline import LogisticRegression

def get_data(split_sequences=False):
    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open('ner.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag = r
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])
            
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    print "number of samples:", len(Xtrain)
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ntest = int(0.3*len(Xtrain))
    Xtest = Xtrain[:Ntest]
    Ytest = Ytrain[:Ntest]
    Xtrain = Xtrain[Ntest:]
    Ytrain = Ytrain[Ntest:]
    print "number of classes:", len(tag2idx)
    return Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx


# def get_data2(split_sequences=False):
#     word2idx = {}
#     tag2idx = {}
#     word_idx = 0
#     tag_idx = 0
#     Xtrain = []
#     Ytrain = []
#     for line in open('../large_files/aij-wikiner-en-wp3'):
#         # each line is a full sentence
#         currentX = []
#         currentY = []
#         line = line.rstrip()
#         if not line:
#             continue
#         triples = line.split()
#         for triple in triples:
#             word, _, tag = triple.split('|')
#             if word not in word2idx:
#                 word2idx[word] = word_idx
#                 word_idx += 1
#             currentX.append(word2idx[word])
            
#             if tag not in tag2idx:
#                 tag2idx[tag] = tag_idx
#                 tag_idx += 1
#             currentY.append(tag2idx[tag])

#         Xtrain.append(currentX)
#         Ytrain.append(currentY)

#     if not split_sequences:
#         Xtrain = np.concatenate(Xtrain)
#         Ytrain = np.concatenate(Ytrain)

#     print "number of samples:", len(Xtrain)
#     Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
#     Ntest = int(0.3*len(Xtrain))
#     Xtest = Xtrain[:Ntest]
#     Ytest = Ytrain[:Ntest]
#     Xtrain = Xtrain[Ntest:]
#     Ytrain = Ytrain[Ntest:]
#     print "number of classes:", len(tag2idx)
#     return Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx


def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx = get_data()

    V = len(word2idx)
    print "vocabulary size:", V
    K = len(tag2idx)

    # train and score
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V, K=K, epochs=5)
    print "training complete"
    print "train score:", model.score(Xtrain, Ytrain)
    print "train f1 score:", model.f1_score(Xtrain, Ytrain)
    print "test score:", model.score(Xtest, Ytest)
    print "test f1 score:", model.f1_score(Xtest, Ytest)

if __name__ == '__main__':
    main()
