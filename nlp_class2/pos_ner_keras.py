# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath('..'))
#from pos_baseline import get_data
from sklearn.utils import shuffle
#from util import init_weight
from datetime import datetime
#from sklearn.metrics import f1_score

from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Input, Dense, Embedding, GRU, LSTM, SimpleRNN #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore


MAX_VOCAB_SIZE = 20000
MAX_TAGS = 100



# def get_data_pos(split_sequences=False):
#   if not os.path.exists('chunking'):
#     print("Please create a folder in your local directory called 'chunking'")
#     print("train.txt and test.txt should be stored in there.")
#     print("Please check the comments to get the download link.")
#     exit()
#   elif not os.path.exists('chunking/train.txt'):
#     print("train.txt is not in chunking/train.txt")
#     print("Please check the comments to get the download link.")
#     exit()
#   elif not os.path.exists('chunking/test.txt'):
#     print("test.txt is not in chunking/test.txt")
#     print("Please check the comments to get the download link.")
#     exit()

#   Xtrain = []
#   Ytrain = []
#   currentX = []
#   currentY = []
#   for line in open('chunking/train.txt', encoding='utf-8'):
#     line = line.rstrip()
#     if line:
#       r = line.split()
#       word, tag, _ = r
#       currentX.append(word)
      
#       currentY.append(tag)
#     elif split_sequences:
#       Xtrain.append(currentX)
#       Ytrain.append(currentY)
#       currentX = []
#       currentY = []

#   if not split_sequences:
#     Xtrain = currentX
#     Ytrain = currentY

#   # load and score test data
#   Xtest = []
#   Ytest = []
#   currentX = []
#   currentY = []
#   for line in open('chunking/test.txt', encoding='utf-8'):
#     line = line.rstrip()
#     if line:
#       r = line.split()
#       word, tag, _ = r
#       currentX.append(word)
#       currentY.append(tag)
#     elif split_sequences:
#       Xtest.append(currentX)
#       Ytest.append(currentY)
#       currentX = []
#       currentY = []
#   if not split_sequences:
#     Xtest = currentX
#     Ytest = currentY

#   return Xtrain, Ytrain, Xtest, Ytest


def get_data_ner(split_sequences=False):
  Xtrain = []
  Ytrain = []
  currentX = []
  currentY = []
  for line in open('ner.txt', encoding='utf-8'):
    line = line.rstrip()
    if line:
      r = line.split()
      word, tag = r
      word = word.lower()
      currentX.append(word)
      currentY.append(tag)
    elif split_sequences:
      Xtrain.append(currentX)
      Ytrain.append(currentY)
      currentX = []
      currentY = []

  if not split_sequences:
    Xtrain = currentX
    Ytrain = currentY

  print("number of samples:", len(Xtrain))
  Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
  Ntest = int(0.3*len(Xtrain))
  Xtest = Xtrain[:Ntest]
  Ytest = Ytrain[:Ntest]
  Xtrain = Xtrain[Ntest:]
  Ytrain = Ytrain[Ntest:]
  return Xtrain, Ytrain, Xtest, Ytest




# get the data
Xtrain, Ytrain, Xtest, Ytest = get_data_ner(split_sequences=True)


# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(Xtrain)
Xtrain = tokenizer.texts_to_sequences(Xtrain)
Xtest  = tokenizer.texts_to_sequences(Xtest)

# get word -> integer mapping
word2idx = tokenizer.word_index
print(f'Found {len(word2idx)} unique tokens.')
vocab_size = min(MAX_VOCAB_SIZE, len(word2idx) + 1)


# convert the tags (strings) into integers
tokenizer2 = Tokenizer(num_words=MAX_TAGS)
tokenizer2.fit_on_texts(Ytrain)
Ytrain = tokenizer2.texts_to_sequences(Ytrain)
Ytest  = tokenizer2.texts_to_sequences(Ytest)

# get tag -> integer mapping
tag2idx = tokenizer2.word_index
print(f'Found {len(tag2idx)} unique tags.')
num_tags = min(MAX_TAGS, len(tag2idx) + 1)


# pad sequences
sequence_length = max(len(x) for x in Xtrain + Xtest)
Xtrain = pad_sequences(Xtrain, maxlen=sequence_length)
Ytrain = pad_sequences(Ytrain, maxlen=sequence_length)
Xtest  = pad_sequences(Xtest,  maxlen=sequence_length)
Ytest  = pad_sequences(Ytest,  maxlen=sequence_length)
print("Xtrain.shape:", Xtrain.shape)
print("Ytrain.shape:", Ytrain.shape)


# one-hot the targets
Ytrain_onehot = np.zeros((len(Ytrain), sequence_length, num_tags), dtype='float32')
for n, sample in enumerate(Ytrain):
  for t, tag in enumerate(sample):
    Ytrain_onehot[n, t, tag] = 1

Ytest_onehot = np.zeros((len(Ytest), sequence_length, num_tags), dtype='float32')
for n, sample in enumerate(Ytest):
  for t, tag in enumerate(sample):
    Ytest_onehot[n, t, tag] = 1



# training config
epochs = 300
batch_size = 32
hidden_layer_size = 10
embedding_dim = 10




# build the model
input_ = Input(shape=(sequence_length,))
x = Embedding(vocab_size, embedding_dim)(input_)
x = SimpleRNN(hidden_layer_size, return_sequences=True)(x)
output = Dense(num_tags, activation='softmax')(x)


model = Model(input_, output)
model.compile(
  loss='categorical_crossentropy',
  optimizer=Adam(learning_rate=1e-2),
  metrics=['accuracy']
)


print('Training model...')
with tf.device('/GPU:0'):
  r = model.fit(Xtrain,
                Ytrain_onehot,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(Xtest, Ytest_onehot))

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

