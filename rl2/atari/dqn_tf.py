# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize





##### testing only
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 1000


MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4 #env.action_space.n


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
        [IM_SIZE, IM_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      self.output = tf.squeeze(self.output)

  def transform(self, state, sess=None):
    sess = sess or tf.get_default_session()
    return sess.run(self.output, { self.input_state: state })


def update_state(state, obs_small):
  return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis=2)



class ReplayMemory:
  def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE, 
               agent_history_length=4, batch_size=32):
    """
    Args:
        size: Integer, Number of stored transitions
        frame_height: Integer, Height of a frame of an Atari game
        frame_width: Integer, Width of a frame of an Atari game
        agent_history_length: Integer, Number of frames stacked together to create a state
        batch_size: Integer, Number of transitions returned in a minibatch
    """
    self.size = size
    self.frame_height = frame_height
    self.frame_width = frame_width
    self.agent_history_length = agent_history_length
    self.batch_size = batch_size
    self.count = 0
    self.current = 0
    
    # Pre-allocate memory
    self.actions = np.empty(self.size, dtype=np.int32)
    self.rewards = np.empty(self.size, dtype=np.float32)
    self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
    self.terminal_flags = np.empty(self.size, dtype=np.bool)
    
    # Pre-allocate memory for the states and new_states in a minibatch
    self.states = np.empty((self.batch_size, self.agent_history_length, 
                            self.frame_height, self.frame_width), dtype=np.uint8)
    self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
    self.indices = np.empty(self.batch_size, dtype=np.int32)
      
  def add_experience(self, action, frame, reward, terminal):
    """
    Args:
        action: An integer-encoded action
        frame: One grayscale frame of the game
        reward: reward the agend received for performing an action
        terminal: A bool stating whether the episode terminated
    """
    if frame.shape != (self.frame_height, self.frame_width):
      raise ValueError('Dimension of frame is wrong!')
    self.actions[self.current] = action
    self.frames[self.current, ...] = frame
    self.rewards[self.current] = reward
    self.terminal_flags[self.current] = terminal
    self.count = max(self.count, self.current+1)
    self.current = (self.current + 1) % self.size
           
  def _get_state(self, index):
    if self.count is 0:
      raise ValueError("The replay memory is empty!")
    if index < self.agent_history_length - 1:
      raise ValueError("Index must be min 3")
    return self.frames[index-self.agent_history_length+1:index+1, ...]
      
  def _get_valid_indices(self):
    for i in range(self.batch_size):
      while True:
        index = random.randint(self.agent_history_length, self.count - 1)
        if index < self.agent_history_length:
          continue
        if index >= self.current and index - self.agent_history_length <= self.current:
          continue
        if self.terminal_flags[index - self.agent_history_length:index].any():
          continue
        break
      self.indices[i] = index
          
  def get_minibatch(self):
    """
    Returns a minibatch of self.batch_size transitions
    """
    if self.count < self.agent_history_length:
      raise ValueError('Not enough memories to get a minibatch')
    
    self._get_valid_indices()
        
    for i, idx in enumerate(self.indices):
      self.states[i] = self._get_state(idx - 1)
      self.new_states[i] = self._get_state(idx)
    
    return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, scope):

    self.K = K
    self.scope = scope

    with tf.variable_scope(scope):

      # inputs and targets
      self.X = tf.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 4), name='X')

      # tensorflow convolution needs the order to be:
      # (num_samples, height, width, "color")

      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

      # calculate output and cost
      # convolutional layers
      # these built-in layers are faster and don't require us to
      # calculate the size of the output of the final conv layer!
      Z = self.X / 255.0
      for num_output_filters, filtersz, poolsz in conv_layer_sizes:
        Z = tf.contrib.layers.conv2d(
          Z,
          num_output_filters,
          filtersz,
          poolsz,
          activation_fn=tf.nn.relu
        )

      # fully connected layers
      Z = tf.contrib.layers.flatten(Z)
      for M in hidden_layer_sizes:
        Z = tf.contrib.layers.fully_connected(Z, M)

      # final output layer
      self.predict_op = tf.contrib.layers.fully_connected(Z, K)

      selected_action_values = tf.reduce_sum(
        self.predict_op * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )

      # cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
      cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_values))
      self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
      # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
      # self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=1e-3).minimize(cost)
      # self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
      # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
      # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

      self.cost = cost


  def copy_from(self, other):
    mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      op = p.assign(q)
      ops.append(op)
    self.session.run(ops)


  def save(self):
    params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    params = self.session.run(params)
    np.savez('tf_dqn_weights.npz', *params)


  def load(self):
    params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    npz = np.load('tf_dqn_weights.npz')
    ops = []
    for p, (_, v) in zip(params, npz.iteritems()):
      ops.append(p.assign(v))
    self.session.run(ops)


  def set_session(self, session):
    self.session = session

  def predict(self, states):
    return self.session.run(self.predict_op, feed_dict={self.X: states})

  def update(self, states, actions, targets):
    c, _ = self.session.run(
      [self.cost, self.train_op],
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
    return c

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
  # Sample experiences
  states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()

  # Calculate targets
  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

  # Update model
  loss = model.update(states, actions, targets)
  return loss


def play_one(
  env,
  sess,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  image_transformer,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  # Reset the environment
  obs = env.reset()
  obs_small = image_transformer.transform(obs, sess)
  state = np.stack([obs_small] * 4, axis=2)
  loss = None


  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  done = False
  while not done:

    # Update target network
    if total_t % TARGET_UPDATE_PERIOD == 0:
      target_model.copy_from(model)
      print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))


    # Take action
    action = model.sample_action(state, epsilon)
    obs, reward, done, _ = env.step(action)
    obs_small = image_transformer.transform(obs, sess)
    next_state = update_state(state, obs_small)

    # Compute total reward
    episode_reward += reward

    # Save the latest experience
    experience_replay_buffer.add_experience(action, obs_small, reward, done)    

    # Train the model, keep track of time
    t0_2 = datetime.now()
    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    dt = datetime.now() - t0_2

    # More debugging info
    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1


    state = next_state
    total_t += 1

    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon


def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y


if __name__ == '__main__':

  # hyperparams and initialize stuff
  conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_layer_sizes = [512]
  gamma = 0.99
  batch_sz = 32
  num_episodes = 3500
  total_t = 0
  experience_replay_buffer = ReplayMemory()
  episode_rewards = np.zeros(num_episodes)



  # epsilon
  # decays linearly until 0.1
  epsilon = 1.0
  epsilon_min = 0.1
  epsilon_change = (epsilon - epsilon_min) / 500000



  # Create environment
  env = gym.envs.make("Breakout-v0")
 


  # Create models
  model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    scope="model")
  target_model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    scope="target_model"
  )
  image_transformer = ImageTransformer()



  with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())


    print("Populating experience replay buffer...")
    obs = env.reset()

    for i in range(MIN_EXPERIENCES):

        action = np.random.choice(K)
        obs, reward, done, _ = env.step(action)
        obs_small = image_transformer.transform(obs, sess) # not used anymore
        experience_replay_buffer.add_experience(action, obs_small, reward, done)

        if done:
            obs = env.reset()


    # Play a number of episodes and learn!
    t0 = datetime.now()
    for i in range(num_episodes):

      total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
        env,
        sess,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
        image_transformer,
        gamma,
        batch_sz,
        epsilon,
        epsilon_change,
        epsilon_min,
      )
      episode_rewards[i] = episode_reward

      last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
      print("Episode:", i,
        "Duration:", duration,
        "Num steps:", num_steps_in_episode,
        "Reward:", episode_reward,
        "Training time per step:", "%.3f" % time_per_step,
        "Avg Reward (Last 100):", "%.3f" % last_100_avg,
        "Epsilon:", "%.3f" % epsilon
      )
      sys.stdout.flush()
    print("Total duration:", datetime.now() - t0)

    model.save()

    # Plot the smoothed returns
    y = smooth(episode_rewards)
    plt.plot(episode_rewards, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()


