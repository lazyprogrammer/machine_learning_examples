import gym
import sys
import os
import numpy as np
import tensorflow as tf

from nets import create_networks


class Step:
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done


# Transform raw images for input into neural network
# 1) Convert to grayscale
# 2) Resize
# 3) Crop
class ImageTransformer:
  def __init__(self):
    with tf.variable_scope("image_transformer"):
      self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
      self.output = tf.image.rgb_to_grayscale(self.input_state)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize_images(
        self.output,
        [84, 84],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      self.output = tf.squeeze(self.output)

  def transform(self, state, sess=None):
    sess = sess or tf.get_default_session()
    return sess.run(self.output, { self.input_state: state })


# Create initial state by repeating the same frame 4 times
def repeat_frame(frame):
  return np.stack([frame] * 4, axis=2)


# Create next state by shifting each frame by 1
# Throw out the oldest frame
# And concatenate the newest frame
def shift_frames(state, next_frame):
  return np.append(state[:,:,1:], np.expand_dims(next_frame, 2), axis=2)


# Make a Tensorflow op to copy weights from one scope to another
def get_copy_params_op(src_vars, dst_vars):
  src_vars = list(sorted(src_vars, key=lambda v: v.name))
  dst_vars = list(sorted(dst_vars, key=lambda v: v.name))

  ops = []
  for s, d in zip(src_vars, dst_vars):
    op = d.assign(s)
    ops.append(op)

  return ops


def make_train_op(local_net, global_net):
  """
  Use gradients from local network to update the global network
  """

  # Idea:
  # We want a list of gradients and corresponding variables
  # e.g. [[g1, g2, g3], [v1, v2, v3]]
  # Since that's what the optimizer expects.
  # But we would like the gradients to come from the local network
  # And the variables to come from the global network
  # So we want to make a list like this:
  # [[local_g1, local_g2, local_g3], [global_v1, global_v2, global_v3]]

  # First get only the gradients
  local_grads, _ = zip(*local_net.grads_and_vars)

  # Clip gradients to avoid large values
  local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)

  # Get global vars
  _, global_vars = zip(*global_net.grads_and_vars)

  # Combine local grads and global vars
  local_grads_global_vars = list(zip(local_grads, global_vars))

  # Run a gradient descent step, e.g.
  # var = var - learning_rate * grad
  return global_net.optimizer.apply_gradients(
    local_grads_global_vars,
    global_step=tf.train.get_global_step())


# Worker object to be run in a thread
# name (String) should be unique for each thread
# env (OpenAI Gym Environment) should be unique for each thread
# policy_net (PolicyNetwork) should be a global passed to every worker
# value_net (ValueNetwork) should be a global passed to every worker
# returns_list (List) should be a global passed to every worker
class Worker:
  def __init__(
      self,
      name,
      env,
      policy_net,
      value_net,
      global_counter,
      returns_list,
      discount_factor=0.99,
      max_global_steps=None):

    self.name = name
    self.env = env
    self.global_policy_net = policy_net
    self.global_value_net = value_net
    self.global_counter = global_counter
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.train.get_global_step()
    self.img_transformer = ImageTransformer()

    # Create local policy and value networks that belong only to this worker
    with tf.variable_scope(name):
      # self.policy_net = PolicyNetwork(num_outputs=policy_net.num_outputs)
      # self.value_net = ValueNetwork()
      self.policy_net, self.value_net = create_networks(policy_net.num_outputs)

    # We will use this op to copy the global network weights
    # back to the local policy and value networks
    self.copy_params_op = get_copy_params_op(
      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global"),
      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/'))

    # These will take the gradients from the local networks
    # and use those gradients to update the global network
    self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
    self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

    self.state = None # Keep track of the current state
    self.total_reward = 0. # After each episode print the total (sum of) reward
    self.returns_list = returns_list # Global returns list to plot later

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Assign the initial state
      self.state = repeat_frame(self.img_transformer.transform(self.env.reset()))

      try:
        while not coord.should_stop():
          # Copy weights from  global networks to local networks
          sess.run(self.copy_params_op)

          # Collect some experience
          steps, global_step = self.run_n_steps(t_max, sess)

          # Stop once the max number of global steps has been reached
          if self.max_global_steps is not None and global_step >= self.max_global_steps:
            coord.request_stop()
            return

          # Update the global networks using local gradients
          self.update(steps, sess)

      except tf.errors.CancelledError:
        return

  def sample_action(self, state, sess):
    # Make input N x D (N = 1)
    feed_dict = { self.policy_net.states: [state] }
    actions = sess.run(self.policy_net.sample_action, feed_dict)
    # Prediction is a 1-D array of length N, just want the first value
    return actions[0]

  def get_value_prediction(self, state, sess):
    # Make input N x D (N = 1)
    feed_dict = { self.value_net.states: [state] }
    vhat = sess.run(self.value_net.vhat, feed_dict)
    # Prediction is a 1-D array of length N, just want the first value
    return vhat[0]

  def run_n_steps(self, n, sess):
    steps = []
    for _ in range(n):
      # Take a step
      action = self.sample_action(self.state, sess)
      next_frame, reward, done, _ = self.env.step(action)

      # Shift the state to include the latest frame
      next_state = shift_frames(self.state, self.img_transformer.transform(next_frame))

      # Save total return
      if done:
        print("Total reward:", self.total_reward, "Worker:", self.name)
        self.returns_list.append(self.total_reward)
        if len(self.returns_list) > 0 and len(self.returns_list) % 100 == 0:
          print("*** Total average reward (last 100):", np.mean(self.returns_list[-100:]), "Collected so far:", len(self.returns_list))
        self.total_reward = 0.
      else:
        self.total_reward += reward

      # Save step
      step = Step(self.state, action, reward, next_state, done)
      steps.append(step)

      # Increase local and global counters
      global_step = next(self.global_counter)

      if done:
        self.state = repeat_frame(self.img_transformer.transform(self.env.reset()))
        break
      else:
        self.state = next_state
    return steps, global_step

  def update(self, steps, sess):
    """
    Updates global policy and value networks using the local networks' gradients
    """

    # In order to accumulate the total return
    # We will use V_hat(s') to predict the future returns
    # But we will use the actual rewards if we have them
    # Ex. if we have s1, s2, s3 with rewards r1, r2, r3
    # Then G(s3) = r3 + V(s4)
    #      G(s2) = r2 + r3 + V(s4)
    #      G(s1) = r1 + r2 + r3 + V(s4)
    reward = 0.0
    if not steps[-1].done:
      reward = self.get_value_prediction(steps[-1].next_state, sess)

    # Accumulate minibatch samples
    states = []
    advantages = []
    value_targets = []
    actions = []

    # loop through steps in reverse order
    for step in reversed(steps):
      reward = step.reward + self.discount_factor * reward
      advantage = reward - self.get_value_prediction(step.state, sess)
      # Accumulate updates
      states.append(step.state)
      actions.append(step.action)
      advantages.append(advantage)
      value_targets.append(reward)

    feed_dict = {
      self.policy_net.states: np.array(states),
      self.policy_net.advantage: advantages,
      self.policy_net.actions: actions,
      self.value_net.states: np.array(states),
      self.value_net.targets: value_targets,
    }

    # Train the global estimators using local gradients
    global_step, pnet_loss, vnet_loss, _, _ = sess.run([
      self.global_step,
      self.policy_net.loss,
      self.value_net.loss,
      self.pnet_train_op,
      self.vnet_train_op,
    ], feed_dict)

    # Theoretically could plot these later
    return pnet_loss, vnet_loss
