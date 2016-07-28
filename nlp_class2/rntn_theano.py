import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from sklearn.utils import shuffle
from util import init_weight, get_ptb_data, display_tree
from datetime import datetime


class RecursiveNN:
    def __init__(self, V, D, K):
        self.V = V
        self.D = D
        self.K = K

    def fit(self, trees, learning_rate=10e-4, mu=0.99, reg=10e-3, epochs=15, activation=T.nnet.relu, train_inner_nodes=False):
        D = self.D
        V = self.V
        K = self.K
        self.f = activation
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

        py_x = T.nnet.softmax(h[:,0,:].dot(self.Wo) + self.bo)

        prediction = T.argmax(py_x, axis=1)
        
        rcost = T.sum([(p*p).mean() for p in self.params])
        if train_inner_nodes:
            cost = -T.mean(T.log(py_x[T.arange(labels.shape[0]), labels])) + rcost
        else:
            cost = -T.mean(T.log(py_x[-1, labels[-1]])) + rcost
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.cost_predict_op = theano.function(
            inputs=[words, left_children, right_children, labels],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        self.train_op = theano.function(
            inputs=[words, left_children, right_children, labels],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        sequence_indexes = range(N)
        if train_inner_nodes:
            n_total = sum(len(words) for words, _, _, _ in trees)
        else:
            n_total = N
        for i in xrange(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            cost = 0
            it = 0
            for j in sequence_indexes:
                words, left, right, lab = trees[j]
                c, p = self.train_op(words, left, right, lab)
                if np.isnan(c):
                    print "Cost is nan! Let's stop here. Why don't you try decreasing the learning rate?"
                    exit()
                cost += c
                if train_inner_nodes:
                    n_correct += np.sum(p == lab)
                else:
                    n_correct += (p[-1] == lab[-1])
                it += 1
                if it % 1 == 0:
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f, cost so far: %f\r" % (it, N, float(n_correct)/n_total, cost))
                    sys.stdout.flush()
            print "i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total), "time for epoch:", (datetime.now() - t0)
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


def add_idx_to_tree(tree, current_idx):
    # post-order labeling of tree nodes
    if tree is None:
        return current_idx
    current_idx = add_idx_to_tree(tree.left, current_idx)
    current_idx = add_idx_to_tree(tree.right, current_idx)
    tree.idx = current_idx
    current_idx += 1
    return current_idx


def tree2list(tree, parent_idx):
    if tree is None:
        return [], [], [], []

    words_left, left_child_left, right_child_left, labels_left = tree2list(tree.left, tree.idx)
    words_right, left_child_right, right_child_right, labels_right = tree2list(tree.right, tree.idx)

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
    labels = labels_left + labels_right + [tree.label]

    return words, left_child, right_child, labels


def main():
    train, test, word2idx = get_ptb_data()

    for t in train:
        add_idx_to_tree(t, 0)
    train = [tree2list(t, -1) for t in train]

    for t in test:
        add_idx_to_tree(t, 0)
    test = [tree2list(t, -1) for t in test]

    train = train[:100]
    test = test[:100]

    V = len(word2idx)
    print "vocab size:", V
    D = 80
    K = 5

    model = RecursiveNN(V, D, K)
    model.fit(train, activation=T.nnet.relu)
    print "train accuracy:", model.score(train)
    print "test accuracy:", model.score(test)


if __name__ == '__main__':
    main()
