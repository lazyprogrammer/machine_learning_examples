# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Model and generate Robert Frost poems.

import numpy as np
import string

initial = {} # start of a phrase
second_word = {}
transitions = {}

def remove_punctuation(s):
    return s.translate(None, string.punctuation)

def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)

for line in open('robert_frost.txt'):
    tokens = remove_punctuation(line.rstrip().lower()).split()

    T = len(tokens)
    for i in xrange(T):
        t = tokens[i]
        if i == 0:
            # measure the distribution of the first word
            initial[t] = initial.get(t, 0.) + 1
        else:
            t_1 = tokens[i-1]
            if i == T - 1:
                # measure probability of ending the line
                add2dict(transitions, (t_1, t), 'END')
            if i == 1:
                # measure distribution of second word
                # given only first word
                add2dict(second_word, t_1, t)
            else:
                t_2 = tokens[i-2]
                add2dict(transitions, (t_2, t_1), t)


# normalize the distributions
initial_total = sum(initial.values())
for t, c in initial.iteritems():
    initial[t] = c / initial_total

def list2pdict(ts):
    # turn each list of possibilities into a dictionary of probabilities
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.iteritems():
        d[t] = c / n
    return d

for t_1, ts in second_word.iteritems():
    # replace list with dictionary of probabilities
    second_word[t_1] = list2pdict(ts)

for k, ts in transitions.iteritems():
    transitions[k] = list2pdict(ts)

# generate 4 lines
def sample_word(d):
    # print "d:", d
    p0 = np.random.random()
    # print "p0:", p0
    cumulative = 0
    for t, p in d.iteritems():
        cumulative += p
        if p0 < cumulative:
            return t
    assert(False) # should never get here

def generate():
    for i in xrange(4):
        sentence =[]

        # initial word
        w0 = sample_word(initial)
        sentence.append(w0)

        # sample second word
        w1 = sample_word(second_word[w0])
        sentence.append(w1)

        # second-order transitions until END
        while True:
            w2 = sample_word(transitions[(w0, w1)])
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2
        print ' '.join(sentence)

generate()

# exercise: make them rhyme!
