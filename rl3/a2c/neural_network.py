# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf


def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def conv(inputs, nf, ks, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs, filters=nf, kernel_size=ks,
                            strides=(strides, strides), activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain))


def dense(inputs, n, act=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs, units=n, activation=act,
                           kernel_initializer=tf.orthogonal_initializer(gain))


class CNN:

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        gain = np.sqrt(2)
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        X_normal = tf.cast(X, tf.float32) / 255.0
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv(X_normal, 32, 8, 4, gain)
            h2 = conv(h1, 64, 4, 2, gain)
            h3 = conv(h2, 64, 3, 1, gain)
            h3 = tf.layers.flatten(h3)
            h4 = dense(h3, 512, gain=gain)
            pi = dense(h4, ac_space.n, act=None)
            vf = dense(h4, 1, act=None)

        v0 = vf[:, 0]
        a0 = sample(pi)
        # self.initial_state = []  # State reserved for LSTM

        def step(ob):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v#, []  # dummy state

        def value(ob):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
