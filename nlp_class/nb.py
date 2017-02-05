# Naive Bayes spam detection for NLP class, which can be found at:
# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python
# dataset: https://archive.ics.uci.edu/ml/datasets/Spambase

# Author: http://lazyprogrammer.me

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# Note: technically multinomial NB is for "counts", but the documentation says
#       it will work for other types of "counts", like tf-idf, so it should
#       also work for our "word proportions"

data = pd.read_csv('spambase.data').as_matrix() # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

X = data[:,:48]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print "Classification rate for NB:", model.score(Xtest, Ytest)



##### you can use ANY model! #####
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print "Classification rate for AdaBoost:", model.score(Xtest, Ytest)