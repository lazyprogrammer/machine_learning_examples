# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from sklearn.utils import shuffle
from util import init_weight, get_ptb_data, display_tree
from datetime import datetime
from sklearn.metrics import f1_score


# helper for adam optimizer
# use tensorflow defaults
# def adam(cost, params, lr0=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
#   grads = T.grad(cost, params)
#   updates = []
#   time = theano.shared(0)
#   new_time = time + 1
#   updates.append((time, new_time))
#   lr = lr0*T.sqrt(1 - beta2**new_time) / (1 - beta1**new_time)
#   for p, g in zip(params, grads):
#     m = theano.shared(p.get_value() * 0.)
#     v = theano.shared(p.get_value() * 0.)
#     new_m = beta1*m + (1 - beta1)*g
#     new_v = beta2*v + (1 - beta2)*g*g
#     new_p = p - lr*new_m / (T.sqrt(new_v) + eps)
#     updates.append((m, new_m))
#     updates.append((v, new_v))
#     updates.append((p, new_p))
#   return updates


# def momentum_updates(cost, params, learning_rate=1e-3, mu=0.99):
#     # momentum changes
#     dparams = [theano.shared(p.get_value() * 0.) for p in params]

#     updates = []
#     grads = T.grad(cost, params)
#     for p, dp, g in zip(params, dparams, grads):
#         dp_update = mu*dp - learning_rate*g
#         p_update = p + dp_update

#         updates.append((dp, dp_update))
#         updates.append((p, p_update))
#     return updates


# def rmsprop(cost, params, lr=1e-3, decay=0.999, eps=1e-10):
#     grads = T.grad(cost, params)
#     caches = [theano.shared(np.ones_like(p.get_value())) for p in params]
#     new_caches = [decay*c + (1. - decay)*g*g for c, g in zip(caches, grads)]

#     c_update = [(c, new_c) for c, new_c in zip(caches, new_caches)]
#     g_update = [
#       (p, p - lr*g / T.sqrt(new_c + eps)) for p, new_c, g in zip(params, new_caches, grads)
#     ]
#     updates = c_update + g_update
#     return updates


def adagrad(cost, params, lr, eps=1e-10):
    grads = T.grad(cost, params)
    caches = [theano.shared(np.ones_like(p.get_value())) for p in params]
    new_caches = [c + g*g for c, g in zip(caches, grads)]

    c_update = [(c, new_c) for c, new_c in zip(caches, new_caches)]
    g_update = [
      (p, p - lr*g / T.sqrt(new_c + eps)) for p, new_c, g in zip(params, new_caches, grads)
    ]
    updates = c_update + g_update
    return updates


class RecursiveNN:
    def __init__(self, V, D, K, activation=T.tanh):
        self.V = V
        self.D = D
        self.K = K
        self.f = activation

    def fit(self, trees, test_trees, reg=1e-3, epochs=8, train_inner_nodes=False):
        D = self.D
        V = self.V
        K = self.K
        N = len(trees)

        We = init_weight(V, D)
        W11 = np.random.randn(D, D, D) / np.sqrt(3*D)
        W22 = np.random.randn(D, D, D) / np.sqrt(3*D)
        W12 = np.random.randn(D, D, D) / np.sqrt(3*D)
        W1 = init_weight(D, D)
        W2 = init_weight(D, D)
        bh = np.zeros(D)
        Wo = init_weight(D, K)
        bo = np.zeros(K)

        self.We = theano.shared(We)
        self.W11 = theano.shared(W11)
        self.W22 = theano.shared(W22)
        self.W12 = theano.shared(W12)
        self.W1 = theano.shared(W1)
        self.W2 = theano.shared(W2)
        self.bh = theano.shared(bh)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.W11, self.W22, self.W12, self.W1, self.W2, self.bh, self.Wo, self.bo]

        lr = T.scalar('learning_rate')
        words = T.ivector('words')
        left_children = T.ivector('left_children')
        right_children = T.ivector('right_children')
        labels = T.ivector('labels')

        def recurrence(n, hiddens, words, left, right):
            w = words[n]
            # any non-word will have index -1
            hiddens = T.switch(
                T.ge(w, 0),
                T.set_subtensor(hiddens[n], self.We[w]),
                T.set_subtensor(hiddens[n],
                    self.f(
                        hiddens[left[n]].dot(self.W11).dot(hiddens[left[n]]) +
                        hiddens[right[n]].dot(self.W22).dot(hiddens[right[n]]) +
                        hiddens[left[n]].dot(self.W12).dot(hiddens[right[n]]) +
                        hiddens[left[n]].dot(self.W1) +
                        hiddens[right[n]].dot(self.W2) +
                        self.bh
                    )
                )
            )
            return hiddens

        hiddens = T.zeros((words.shape[0], D))

        h, _ = theano.scan(
            fn=recurrence,
            outputs_info=[hiddens],
            n_steps=words.shape[0],
            sequences=T.arange(words.shape[0]),
            non_sequences=[words, left_children, right_children],
        )

        py_x = T.nnet.softmax(h[-1].dot(self.Wo) + self.bo)

        prediction = T.argmax(py_x, axis=1)
        
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        if train_inner_nodes:
            relevant_labels = labels[labels >= 0]
            cost = -T.mean(T.log(py_x[labels >= 0, relevant_labels])) + rcost
        else:
            cost = -T.mean(T.log(py_x[-1, labels[-1]])) + rcost
        

        updates = adagrad(cost, self.params, lr)

        self.cost_predict_op = theano.function(
            inputs=[words, left_children, right_children, labels],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        self.train_op = theano.function(
            inputs=[words, left_children, right_children, labels, lr],
            outputs=[cost, prediction],
            updates=updates
        )

        lr_ = 8e-3 # initial learning rate
        costs = []
        sequence_indexes = range(N)
        # if train_inner_nodes:
        #     n_total = sum(len(words) for words, _, _, _ in trees)
        # else:
        #     n_total = N
        for i in range(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            n_total = 0
            cost = 0
            it = 0
            for j in sequence_indexes:
                words, left, right, lab = trees[j]
                c, p = self.train_op(words, left, right, lab, lr_)
                if np.isnan(c):
                    print("Cost is nan! Let's stop here. \
                        Why don't you try decreasing the learning rate?")
                    for p in self.params:
                        print(p.get_value().sum())
                    exit()
                cost += c
                n_correct += (p[-1] == lab[-1])
                n_total += 1
                it += 1
                if it % 10 == 0:
                    sys.stdout.write(
                        "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
                        (it, N, float(n_correct)/n_total, cost)
                    )
                    sys.stdout.flush()

            # calculate the test score
            n_test_correct = 0
            n_test_total = 0
            for words, left, right, lab in test_trees:
                _, p = self.cost_predict_op(words, left, right, lab)
                n_test_correct += (p[-1] == lab[-1])
                n_test_total += 1

            print(
                "i:", i, "cost:", cost,
                "train acc:", float(n_correct)/n_total,
                "test acc:", float(n_test_correct)/n_test_total,
                "time for epoch:", (datetime.now() - t0)
            )
            costs.append(cost)

        plt.plot(costs)
        plt.show()

    def score(self, trees):
        n_total = len(trees)
        n_correct = 0
        for words, left, right, lab in trees:
            _, p = self.cost_predict_op(words, left, right, lab)
            n_correct += (p[-1] == lab[-1])
        return float(n_correct) / n_total

    def f1_score(self, trees):
        Y = []
        P = []
        for words, left, right, lab in trees:
            _, p = self.cost_predict_op(words, left, right, lab)
            Y.append(lab[-1])
            P.append(p[-1])
        return f1_score(Y, P, average=None).mean()


def add_idx_to_tree(tree, current_idx):
    # post-order labeling of tree nodes
    if tree is None:
        return current_idx
    current_idx = add_idx_to_tree(tree.left, current_idx)
    current_idx = add_idx_to_tree(tree.right, current_idx)
    tree.idx = current_idx
    current_idx += 1
    return current_idx


def tree2list(tree, parent_idx, is_binary=False):
    if tree is None:
        return [], [], [], []

    words_left, left_child_left, right_child_left, labels_left = tree2list(tree.left, tree.idx, is_binary)
    words_right, left_child_right, right_child_right, labels_right = tree2list(tree.right, tree.idx, is_binary)

    if tree.word is None:
        w = -1
        left = tree.left.idx
        right = tree.right.idx
    else:
        w = tree.word
        left = -1
        right = -1

    words = words_left + words_right + [w]
    left_child = left_child_left + left_child_right + [left]
    right_child = right_child_left + right_child_right + [right]

    if is_binary:
        if tree.label > 2:
            label = 1
        elif tree.label < 2:
        # else:
            label = 0
        else:
            label = -1 # we will eventually filter these out
    else:
        label = tree.label
    labels = labels_left + labels_right + [label]

    return words, left_child, right_child, labels


def main(is_binary=True):
    train, test, word2idx = get_ptb_data()

    for t in train:
        add_idx_to_tree(t, 0)
    train = [tree2list(t, -1, is_binary) for t in train]
    if is_binary:
        train = [t for t in train if t[3][-1] >= 0] # for filtering binary labels

    for t in test:
        add_idx_to_tree(t, 0)
    test = [tree2list(t, -1, is_binary) for t in test]
    if is_binary:
        test = [t for t in test if t[3][-1] >= 0] # for filtering binary labels

    # check imbalance
    # pos = 0
    # neg = 0
    # mid = 0
    # label_counts = np.zeros(5)
    # for t in train + test:
    #     words, left_child, right_child, labels = t
    #     # for l in labels:
    #     #     if l == 0:
    #     #         neg += 1
    #     #     elif l == 1:
    #     #         pos += 1
    #     #     else:
    #     #         mid += 1
    #     for l in labels:
    #         label_counts[l] += 1
    # # print("pos / total:", float(pos) / (pos + neg + mid))
    # # print("mid / total:", float(mid) / (pos + neg + mid))
    # # print("neg / total:", float(neg) / (pos + neg + mid))
    # print("label proportions:", label_counts / label_counts.sum())
    # exit()


    train = shuffle(train)
    # train = train[:5000]
    # n_pos = sum(t[3][-1] for t in train)
    # print("n_pos train:", n_pos)
    test = shuffle(test)
    smalltest = test[:1000]
    # n_pos = sum(t[3][-1] for t in test)
    # print("n_pos test:", n_pos)

    V = len(word2idx)
    print("vocab size:", V)
    D = 20
    K = 2 if is_binary else 5

    model = RecursiveNN(V, D, K)
    model.fit(train, smalltest, epochs=20, train_inner_nodes=True)
    print("train accuracy:", model.score(train))
    print("test accuracy:", model.score(test))
    print("train f1:", model.f1_score(train))
    print("test f1:", model.f1_score(test))


if __name__ == '__main__':
    main()
