# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
# from util import find_analogies

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances


from glob import glob

import os
import sys
import string



# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3


def download_text8(dst):
  pass


def get_text8():
  # download the data if it is not yet in the right place
  path = '../large_files/text8'
  if not os.path.exists(path):
    download_text8(path)

  words = open(path).read()
  word2idx = {}
  sents = [[]]
  count = 0
  for word in words.split():
    if word not in word2idx:
      word2idx[word] = count
      count += 1
    sents[0].append(word2idx[word])
  print("count:", count)
  return sents, word2idx


def get_wiki():
  V = 20000
  files = glob('../large_files/enwiki*.txt')
  all_word_counts = {}
  for f in files:
    for line in open(f):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
          for word in s:
            if word not in all_word_counts:
              all_word_counts[word] = 0
            all_word_counts[word] += 1
  print("finished counting")

  V = min(V, len(all_word_counts))
  all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

  top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
  word2idx = {w:i for i, w in enumerate(top_words)}
  unk = word2idx['<UNK>']

  sents = []
  for f in files:
    for line in open(f):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
          # if a word is not nearby another word, there won't be any context!
          # and hence nothing to train!
          sent = [word2idx[w] if w in word2idx else unk for w in s]
          sents.append(sent)
  return sents, word2idx




def train_model(savedir):
  # get the data
  sentences, word2idx = get_wiki() #get_text8()


  # number of unique words
  vocab_size = len(word2idx)


  # config
  window_size = 10
  learning_rate = 0.025
  final_learning_rate = 0.0001
  num_negatives = 5 # number of negative samples to draw per input word
  samples_per_epoch = int(1e5)
  epochs = 20
  D = 50 # word embedding size

  # learning rate decay
  learning_rate_delta = (learning_rate - final_learning_rate) / epochs

  # distribution for drawing negative samples
  p_neg = get_negative_sampling_distribution(sentences)


  # params
  W = np.random.randn(vocab_size, D).astype(np.float32) # input-to-hidden
  V = np.random.randn(D, vocab_size).astype(np.float32) # hidden-to-output


  # create the model
  tf_input = tf.placeholder(tf.int32, shape=(None,))
  tf_negword = tf.placeholder(tf.int32, shape=(None,))
  tf_context = tf.placeholder(tf.int32, shape=(None,)) # targets (context)
  tfW = tf.Variable(W)
  tfV = tf.Variable(V.T)
  # biases = tf.Variable(np.zeros(vocab_size, dtype=np.float32))

  def dot(A, B):
    C = A * B
    return tf.reduce_sum(C, axis=1)

  # correct middle word output
  emb_input = tf.nn.embedding_lookup(tfW, tf_input) # 1 x D
  emb_output = tf.nn.embedding_lookup(tfV, tf_context) # N x D
  correct_output = dot(emb_input, emb_output) # N
  # emb_input = tf.transpose(emb_input, (1, 0))
  # correct_output = tf.matmul(emb_output, emb_input)
  pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones(tf.shape(correct_output)), logits=correct_output)

  # incorrect middle word output
  emb_input = tf.nn.embedding_lookup(tfW, tf_negword)
  incorrect_output = dot(emb_input, emb_output)
  # emb_input = tf.transpose(emb_input, (1, 0))
  # incorrect_output = tf.matmul(emb_output, emb_input)
  neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros(tf.shape(incorrect_output)), logits=incorrect_output)

  # total loss
  loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)

  # output = hidden.dot(tfV)

  # loss
  # neither of the built-in TF functions work well
  # per_sample_loss = tf.nn.nce_loss(
  # # per_sample_loss = tf.nn.sampled_softmax_loss(
  #   weights=tfV,
  #   biases=biases,
  #   labels=tfY,
  #   inputs=hidden,
  #   num_sampled=num_negatives,
  #   num_classes=vocab_size,
  # )
  # loss = tf.reduce_mean(per_sample_loss)

  # optimizer
  # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
  # train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

  # make session
  session = tf.Session()
  init_op = tf.global_variables_initializer()
  session.run(init_op)


  # save the costs to plot them per iteration
  costs = []


  # number of total words in corpus
  total_words = sum(len(sentence) for sentence in sentences)
  print("total number of words in corpus:", total_words)


  # for subsampling each sentence
  threshold = 1e-5
  p_drop = 1 - np.sqrt(threshold / p_neg)


  # train the model
  for epoch in range(epochs):
    # randomly order sentences so we don't always see
    # sentences in the same order
    np.random.shuffle(sentences)

    # accumulate the cost
    cost = 0
    counter = 0
    inputs = []
    targets = []
    negwords = []
    t0 = datetime.now()
    for sentence in sentences:

      # keep only certain words based on p_neg
      sentence = [w for w in sentence \
        if np.random.random() < (1 - p_drop[w])
      ]
      if len(sentence) < 2:
        continue


      # randomly order words so we don't always see
      # samples in the same order
      randomly_ordered_positions = np.random.choice(
        len(sentence),
        # size=np.random.randint(1, len(sentence) + 1),
        size=len(sentence),
        replace=False,
      )


      for j, pos in enumerate(randomly_ordered_positions):
        # the middle word
        word = sentence[pos]

        # get the positive context words/negative samples
        context_words = get_context(pos, sentence, window_size)
        neg_word = np.random.choice(vocab_size, p=p_neg)

        
        n = len(context_words)
        inputs += [word]*n
        negwords += [neg_word]*n
        # targets = np.concatenate([targets, targets_])
        targets += context_words

        # _, c = session.run(
        #   (train_op, loss),
        #   feed_dict={
        #     tf_input: [word],
        #     tf_negword: [neg_word],
        #     tf_context: targets_,
        #   }
        # )
        # cost += c


      if len(inputs) >= 128:
        _, c = session.run(
          (train_op, loss),
          feed_dict={
            tf_input: inputs,
            tf_negword: negwords,
            tf_context: targets,
          }
        )
        cost += c

        # reset
        inputs = []
        targets = []
        negwords = []

      counter += 1
      if counter % 100 == 0:
        sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
        sys.stdout.flush()
        # break


    # print stuff so we don't stare at a blank screen
    dt = datetime.now() - t0
    print("epoch complete:", epoch, "cost:", cost, "dt:", dt)

    # save the cost
    costs.append(cost)

    # update the learning rate
    learning_rate -= learning_rate_delta


  # plot the cost per iteration
  plt.plot(costs)
  plt.show()

  # get the params
  W, VT = session.run((tfW, tfV))
  V = VT.T

  # save the model
  if not os.path.exists(savedir):
    os.mkdir(savedir)

  with open('%s/word2idx.json' % savedir, 'w') as f:
    json.dump(word2idx, f)

  np.savez('%s/weights.npz' % savedir, W, V)

  # return the model
  return word2idx, W, V


def get_negative_sampling_distribution(sentences):
  # Pn(w) = prob of word occuring
  # we would like to sample the negative samples
  # such that words that occur more often
  # should be sampled more often

  word_freq = {}
  word_count = sum(len(sentence) for sentence in sentences)
  for sentence in sentences:
      for word in sentence:
          if word not in word_freq:
              word_freq[word] = 0
          word_freq[word] += 1
  
  # vocab size
  V = len(word_freq)

  p_neg = np.zeros(V)
  for j in range(V):
      p_neg[j] = word_freq[j]**0.75

  # normalize it
  p_neg = p_neg / p_neg.sum()

  assert(np.all(p_neg > 0))
  return p_neg


def get_context(pos, sentence, window_size):
  # input:
  # a sentence of the form: x x x x c c c pos c c c x x x x
  # output:
  # the context word indices: c c c c c c

  start = max(0, pos - window_size)
  end_  = min(len(sentence), pos + window_size)

  context = []
  for ctx_pos, ctx_word_idx in enumerate(sentence[start:end_], start=start):
    if ctx_pos != pos:
      # don't include the input word itself as a target
      context.append(ctx_word_idx)
  return context



def load_model(savedir):
  with open('%s/word2idx.json' % savedir) as f:
    word2idx = json.load(f)
  npz = np.load('%s/weights.npz' % savedir)
  W = npz['arr_0']
  V = npz['arr_1']
  return word2idx, W, V



def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
  V, D = W.shape

  # don't actually use pos2 in calculation, just print what's expected
  print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
  for w in (pos1, neg1, pos2, neg2):
    if w not in word2idx:
      print("Sorry, %s not in word2idx" % w)
      return

  p1 = W[word2idx[pos1]]
  n1 = W[word2idx[neg1]]
  p2 = W[word2idx[pos2]]
  n2 = W[word2idx[neg2]]

  vec = p1 - n1 + n2

  distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
  idx = distances.argsort()[:10]

  # pick one that's not p1, n1, or n2
  best_idx = -1
  keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
  for i in idx:
    if i not in keep_out:
      best_idx = i
      break

  print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[idx[0]], neg2))
  print("closest 10:")
  for i in idx:
    print(idx2word[i], distances[i])

  print("dist to %s:" % pos2, cos_dist(p2, vec))


def test_model(word2idx, W, V):
  # there are multiple ways to get the "final" word embedding
  # We = (W + V.T) / 2
  # We = W

  idx2word = {i:w for w, i in word2idx.items()}

  for We in (W, (W + V.T) / 2):
    print("**********")

    analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
    analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
    analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
    analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
    analogy('japan', 'sushi', 'england', 'bread', word2idx, idx2word, We)
    analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
    analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
    analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
    analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
    analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
    analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
    analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
    analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
    analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
    analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
    analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
    analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
    analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
    analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
    analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
    analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
    analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
    analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
    analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)



if __name__ == '__main__':
  word2idx, W, V = train_model('w2v_tf')
  # word2idx, W, V = load_model('w2v_tf')
  test_model(word2idx, W, V)

