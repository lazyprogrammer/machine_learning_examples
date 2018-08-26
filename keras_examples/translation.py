# https://deeplearningcourses.com/c/deep-learning-advanced-nlp
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU




# some configuration
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 20
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 30
NUM_SAMPLES = 10000



# Where we will store the data
input_texts = [] # sentence in original language
target_texts = [] # sentence in target language


# load in the data
# download the data at: http://www.manythings.org/anki/
t = 0
for line in open('../large_files/translation/spa.txt'):
  # only keep a limited number of samples
  t += 1
  if t > NUM_SAMPLES:
    break

  # input and target are separated by tab
  if '\t' not in line:
    continue

  # split up the input and translation
  input_text, translation = line.rstrip().split('\t')

  input_texts.append(input_text)
  target_texts.append(translation)
print("num samples:", len(input_texts))



# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))
num_words_input = len(word2idx_inputs) + 1

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(target_texts)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)

# get the word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)


max_len_both = max(max_len_input, max_len_target)



# pad the sequences
inputs_padded = pad_sequences(input_sequences, maxlen=max_len_both)
targets_padded = pad_sequences(target_sequences, maxlen=max_len_both)



# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
targets_padded_one_hot = np.zeros(
  (
    len(targets_padded),
    max_len_both,
    num_words_output
  ),
  dtype='float32'
)

# assign the values
for i, d in enumerate(targets_padded):
  for t, word in enumerate(d):
    targets_padded_one_hot[i, t, word] = 1




print('Building model...')

# create an LSTM network with a single LSTM
input_ = Input(shape=(max_len_both,))
x = Embedding(num_words_input, EMBEDDING_DIM)(input_)
x = Bidirectional(LSTM(15, return_sequences=True))(x)
output = Dense(num_words_output, activation='softmax')(x)

model = Model(input_, output)
model.compile(
  loss='categorical_crossentropy',
  optimizer=Adam(lr=0.1),
  metrics=['accuracy']
)


print('Training model...')
r = model.fit(
  inputs_padded,
  targets_padded_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
