import tensorflow as tf


def build_feature_extractor(input_):
  # We only want to create the weights once
  # In all future calls we should set reuse = True

  # scale the inputs from 0..255 to 0..1
  input_ = tf.to_float(input_) / 255.0

  # conv layers
  conv1 = tf.contrib.layers.conv2d(
    input_,
    16, # num output feature maps
    8,  # kernel size
    4,  # stride
    activation_fn=tf.nn.relu,
    scope="conv1")
  conv2 = tf.contrib.layers.conv2d(
    conv1,
    32, # num output feature maps
    4,  # kernel size
    2,  # stride
    activation_fn=tf.nn.relu,
    scope="conv2")

  # image -> feature vector
  flat = tf.contrib.layers.flatten(conv2)

  # dense layer
  fc1 = tf.contrib.layers.fully_connected(
    inputs=flat,
    num_outputs=256,
    scope="fc1")

  return fc1

class PolicyNetwork:
  def __init__(self, num_outputs, reg=0.01):
    self.num_outputs = num_outputs

    # Graph inputs
    # After resizing we have 4 consecutive frames of size 84 x 84
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    # Advantage = G - V(s)
    self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
    # Selected actions
    self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

    # Since we set reuse=False here, that means we MUST
    # create the PolicyNetwork before creating the ValueNetwork
    # ValueNetwork will use reuse=True
    with tf.variable_scope("shared", reuse=False):
      fc1 = build_feature_extractor(self.states)

    # Use a separate scope for output and loss
    with tf.variable_scope("policy_network"):
      self.logits = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
      self.probs = tf.nn.softmax(self.logits)

      # Sample an action
      cdist = tf.distributions.Categorical(logits=self.logits)
      self.sample_action = cdist.sample()

      # Add regularization to increase exploration
      self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), axis=1)

      # Get the predictions for the chosen actions only
      batch_size = tf.shape(self.states)[0]
      gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
      self.selected_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

      self.loss = tf.log(self.selected_action_probs) * self.advantage + reg * self.entropy
      self.loss = -tf.reduce_sum(self.loss, name="loss")

      # training
      self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

      # we'll need these later for running gradient descent steps
      self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
      self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


class ValueNetwork:
  def __init__(self):
    # Placeholders for our input
    # After resizing we have 4 consecutive frames of size 84 x 84
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    # The TD target value
    self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

    # Since we set reuse=True here, that means we MUST
    # create the PolicyNetwork before creating the ValueNetwork
    # PolictyNetwork will use reuse=False
    with tf.variable_scope("shared", reuse=True):
      fc1 = build_feature_extractor(self.states)

    # Use a separate scope for output and loss
    with tf.variable_scope("value_network"):
      self.vhat = tf.contrib.layers.fully_connected(
        inputs=fc1,
        num_outputs=1,
        activation_fn=None)
      self.vhat = tf.squeeze(self.vhat, squeeze_dims=[1], name="vhat")

      self.loss = tf.squared_difference(self.vhat, self.targets)
      self.loss = tf.reduce_sum(self.loss, name="loss")

      # training
      self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

      # we'll need these later for running gradient descent steps
      self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
      self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


# Should use this to create networks
# to ensure they're created in the correct order
def create_networks(num_outputs):
  policy_network = PolicyNetwork(num_outputs=num_outputs)
  value_network = ValueNetwork()
  return policy_network, value_network
