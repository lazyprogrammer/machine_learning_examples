# This code is for my NLP Udemy class, which can be found at:
# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python
# It is written in such a way that tells a story.
# i.e. So you can follow a thought process of starting from a
# simple idea, hitting an obstacle, overcoming it, etc.
# i.e. It is not optimized for anything.

# Author: http://lazyprogrammer.me

import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

# there are more positive reviews than negative reviews
# so let's take a random sample so we have balanced classes
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

# first let's just try to tokenize the text using nltk's tokenizer
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)
#
# notice how it doesn't downcase, so It != it
# not only that, but do we really want to include the word "it" anyway?
# you can imagine it wouldn't be any more common in a positive review than a negative review
# so it might only add noise to our model.
# so let's create a function that does all this pre-processing for us

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


# now let's create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print "Classification rate:", model.score(Xtest, Ytest)


# let's look at the weights for each word
# try it with different threshold values!
threshold = 0.5
for word, index in word_index_map.iteritems():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print word, weight
