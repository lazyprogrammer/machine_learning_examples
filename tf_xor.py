import tensorflow as tf
import numpy as np

### xor ###
# trX = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# trY = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])


### donut ###
N = 1000
R_inner = 5
R_outer = 10
ones = np.array([[1]*N]).T

# distance from origin is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N/2) + R_inner
theta = 2*np.pi*np.random.random(N/2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N/2) + R_outer
theta = 2*np.pi*np.random.random(N/2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

trX = np.concatenate([ X_inner, X_outer ])
trX = np.concatenate((ones, trX), axis=1)
trY = np.array([[0, 1]]*(N/2) + [[1, 0]]*(N/2))





def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_drop_hidden)

    return tf.matmul(h2, w_o)


X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 2])

w_h = init_weights([3, 8])
w_h2 = init_weights([8, 2])
w_o = init_weights([2, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.00001, 1.0).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


for i in range(1000):
    # for 
    sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: 0.8, p_keep_hidden: 0.5})

    print i, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY,
                                                     p_keep_input: 0.8,
                                                     p_keep_hidden: 0.5}))



