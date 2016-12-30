# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
# data is from: http://nlp.stanford.edu/sentiment/

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_ptb_data, display_tree


def tensor_mul(d, x1, A, x2):
    A = tf.reshape(A, [d, d*d])
    # (1 x d) x (d x dd)
    tmp = tf.matmul(x1, A)
    # (1 x dd)
    tmp = tf.reshape(tmp, [d, d])
    # (d x d)
    tmp = tf.matmul(tmp, tf.transpose(x2))
    # (d x 1)
    return tf.reshape(tmp, [1, d])


def get_labels(tree):
    # must be returned in the same order as tree logits are returned
    # post-order traversal
    if tree is None:
        return []
    return get_labels(tree.left) + get_labels(tree.right) + [tree.label]


class RNTN:
    def __init__(self, V, D, K, activation):
        self.D = D
        self.f = activation

        # word embedding
        We = init_weight(V, D)

        # quadratic terms
        W11 = np.random.randn(D, D, D) / np.sqrt(3*D)
        W22 = np.random.randn(D, D, D) / np.sqrt(3*D)
        W12 = np.random.randn(D, D, D) / np.sqrt(3*D)

        # linear terms
        W1 = init_weight(D, D)
        W2 = init_weight(D, D)

        # bias
        bh = np.zeros(D)

        # output layer
        Wo = init_weight(D, K)
        bo = np.zeros(K)

        # make them tensorflow variables
        self.We = tf.Variable(We.astype(np.float32))
        self.W11 = tf.Variable(W11.astype(np.float32))
        self.W22 = tf.Variable(W22.astype(np.float32))
        self.W12 = tf.Variable(W12.astype(np.float32))
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.bh = tf.Variable(bh.astype(np.float32))
        self.Wo = tf.Variable(Wo.astype(np.float32))
        self.bo = tf.Variable(bo.astype(np.float32))
        self.params = [self.We, self.W11, self.W22, self.W12, self.W1, self.W2, self.Wo]

    def fit(self, trees, lr=10e-3, mu=0.9, reg=10e-2, epochs=5):
        train_ops = []
        costs = []
        predictions = []
        all_labels = []
        i = 0
        N = len(trees)
        print "Compiling ops"
        for t in trees:
            i += 1
            sys.stdout.write("%d/%d\r" % (i, N))
            sys.stdout.flush()
            logits = self.get_output(t)
            labels = get_labels(t)
            all_labels.append(labels)

            cost = self.get_cost(logits, labels, reg)
            costs.append(cost)

            prediction = tf.argmax(logits, 1)
            predictions.append(prediction)

            train_op = tf.train.MomentumOptimizer(lr, mu).minimize(cost)
            train_ops.append(train_op)

        # save for later so we don't have to recompile if we call score
        self.predictions = predictions
        self.all_labels = all_labels

        self.saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        actual_costs = []
        per_epoch_costs = []
        correct_rates = []
        with tf.Session() as session:
            session.run(init)

            for i in xrange(epochs):
                train_ops, costs, predictions, all_labels = shuffle(train_ops, costs, predictions, all_labels)
                epoch_cost = 0
                n_correct = 0
                n_total = 0
                j = 0
                N = len(train_ops)
                for train_op, cost, prediction, labels in zip(train_ops, costs, predictions, all_labels):
                    _, c, p = session.run([train_op, cost, prediction])
                    epoch_cost += c
                    actual_costs.append(c)
                    n_correct += np.sum(p == labels)
                    n_total += len(labels)

                    j += 1
                    if j % 10 == 0:
                        sys.stdout.write("j: %d, N: %d, c: %f\r" % (j, N, c))
                        sys.stdout.flush()

                    if np.isnan(c):
                        exit()

                per_epoch_costs.append(epoch_cost)
                correct_rates.append(n_correct / float(n_total))

            self.save_path = self.saver.save(session, "tf_model.ckpt")

        plt.plot(actual_costs)
        plt.title("cost per train_op call")
        plt.show()

        plt.plot(per_epoch_costs)
        plt.title("per epoch costs")
        plt.show()

        plt.plot(correct_rates)
        plt.title("correct rates")
        plt.show()

    def get_cost(self, logits, labels, reg):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
        rcost = sum(tf.nn.l2_loss(p) for p in self.params)
        cost += reg*rcost
        return cost

    # list_of_logits is an output!
    # it is added to using post-order traversal
    def get_output_recursive(self, tree, list_of_logits, is_root=True):
        if tree.word is not None:
            # this is a leaf node
            x = tf.nn.embedding_lookup(self.We, [tree.word])
        else:
            # if tree.left is None or tree.right is None:
            #     print "This tree node has no word but also has no children:"
            #     display_tree(tree)
            #     print ""
            # this node has children
            x1 = self.get_output_recursive(tree.left, list_of_logits, is_root=False)
            x2 = self.get_output_recursive(tree.right, list_of_logits, is_root=False)
            x = self.f(
                tensor_mul(self.D, x1, self.W11, x1) +
                tensor_mul(self.D, x2, self.W22, x2) +
                tensor_mul(self.D, x1, self.W12, x2) +
                tf.matmul(x1, self.W1) +
                tf.matmul(x2, self.W2) +
                self.bh)

        logits = tf.matmul(x, self.Wo) + self.bo
        list_of_logits.append(logits)

        return x

    def get_output(self, tree):
        logits = []
        try:
            self.get_output_recursive(tree, logits)
        except Exception as e:
            display_tree(tree)
            raise e
        return tf.concat(0, logits)

    def score(self, trees):
        if trees is None:
            predictions = self.predictions
            all_labels = self.all_labels
        else:
            # just build and run the predict_op for each tree
            # and accumulate the total
            predictions = []
            all_labels = []

            i = 0
            N = len(trees)
            print "Compiling ops"
            for t in trees:

                i += 1
                sys.stdout.write("%d/%d\r" % (i, N))
                sys.stdout.flush()

                logits = self.get_output(t)
                labels = get_labels(t)
                all_labels.append(labels)

                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

        n_correct = 0
        n_total = 0
        with tf.Session() as session:
            self.saver.restore(session, "tf_model.ckpt")

            for prediction, y in zip(predictions, all_labels):
                p = session.run(prediction)
                # print "pred:", p
                # print "label:", y
                # n_correct += np.sum(p == y)
                n_correct += (p[-1] == y[-1]) # we only care about the root
                n_total += len(y)

        return float(n_correct) / n_total


def main():
    train, test, word2idx = get_ptb_data()

    train = train[:100]
    test = test[:100]

    V = len(word2idx)
    D = 80
    K = 5

    model = RNTN(V, D, K, tf.nn.relu)
    model.fit(train)
    print "train accuracy:", model.score(None)
    print "test accuracy:", model.score(test)


if __name__ == '__main__':
    main()
