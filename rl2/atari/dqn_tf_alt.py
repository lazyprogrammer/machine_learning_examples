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
IM_SIZE = 80
K = 4 #env.action_space.n




def downsample_image(A):
  B = A[31:195] # select the important parts of the image
  B = B.mean(axis=2) # convert to grayscale

  # downsample image
  # changing aspect ratio doesn't significantly distort the image
  # nearest neighbor interpolation produces a much sharper image
  # than default bilinear
  B = imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')
  return B


def update_state(state, obs):
  obs_small = downsample_image(obs)
  return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)





class ConvLayer:
  def __init__(self, mi, mo, filtersz=5, stride=2, f=tf.nn.relu):
    # mi = input feature map size
    # mo = output feature map size
    self.W = tf.Variable(tf.random_normal(shape=(filtersz, filtersz, mi, mo)))
    b0 = np.zeros(mo, dtype=np.float32)
    self.b = tf.Variable(b0)
    self.f = f
    self.stride = stride
    self.params = [self.W, self.b]

  def forward(self, X):
    conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, self.b)
    return self.f(conv_out)


class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.relu, use_bias=True):
    # print("M1:", M1)
    self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
      self.params.append(self.b)
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


class DQN:
  # def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, max_experiences=500000, min_experiences=50000, batch_sz=32):
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma):
    self.K = K

    # create the graph
    self.conv_layers = []
    num_input_filters = 4 # number of filters / color channels
    final_height = IM_SIZE
    final_width = IM_SIZE
    for num_output_filters, filtersz, stride in conv_layer_sizes:
      layer = ConvLayer(num_input_filters, num_output_filters, filtersz, stride)
      self.conv_layers.append(layer)
      num_input_filters = num_output_filters

      # calculate final output size for input into fully connected layers
      old_height = final_height
      new_height = int(np.ceil(old_height / stride))
      print("new_height (%s) = old_height (%s) / stride (%s)" % (new_height, old_height, stride))
      final_height = int(np.ceil(final_height / stride))
      final_width = int(np.ceil(final_width / stride))

    self.layers = []
    flattened_ouput_size = final_height * final_width * num_input_filters
    M1 = flattened_ouput_size
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, lambda x: x)
    self.layers.append(layer)

    # collect params for copy
    self.params = []
    for layer in (self.conv_layers + self.layers):
      self.params += layer.params

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, 4, IM_SIZE, IM_SIZE), name='X')
    # tensorflow convolution needs the order to be:
    # (num_samples, height, width, "color")
    # so we need to tranpose later
    self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

    # calculate output and cost
    Z = self.X / 255.0
    Z = tf.transpose(Z, [0, 2, 3, 1]) # TF wants the "color" channel to be last
    for layer in self.conv_layers:
      Z = layer.forward(Z)
    Z = tf.reshape(Z, [-1, flattened_ouput_size])
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = Z
    self.predict_op = Y_hat

    # selected_action_values = tf.reduce_sum(
    #   Y_hat * tf.one_hot(self.actions, K),
    #   reduction_indices=[1]
    # )

    # we would like to do this, but it doesn't work in TF:
    # selected_action_values = Y_hat[tf.range(batch_sz), self.actions]
    # instead we do:
    indices = tf.range(batch_sz) * tf.shape(Y_hat)[1] + self.actions
    selected_action_values = tf.gather(
      tf.reshape(Y_hat, [-1]), # flatten
      indices
    )

    cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
    self.cost = cost
    # self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
    # self.train_op = tf.train.AdagradOptimizer(10e-3).minimize(cost)
    self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(10e-4, momentum=0.9).minimize(cost)
    # self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)

  def set_session(self, session):
    self.session = session

  def copy_from(self, other):
    # collect all the ops
    ops = []
    my_params = self.params
    other_params = other.params
    for p, q in zip(my_params, other_params):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)
    # now run them all
    self.session.run(ops)

  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.X: X})

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
  samples = random.sample(experience_replay_buffer, batch_size)
  states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

  # Calculate targets
  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

  # Update model
  loss = model.update(states, actions, targets)
  return loss


def play_one(
  env,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  # Reset the environment
  obs = env.reset()
  obs_small = downsample_image(obs)
  state = np.stack([obs_small] * 4, axis=0)
  assert(state.shape == (4, 80, 80))
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
    obs_small = downsample_image(obs)
    next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
    # assert(state.shape == (4, 80, 80))



    episode_reward += reward

    # Remove oldest experience if replay buffer is full
    if len(experience_replay_buffer) == MAX_EXPERIENCES:
      experience_replay_buffer.pop(0)

    # Save the latest experience
    experience_replay_buffer.append((state, action, reward, next_state, done))

    # Train the model, keep track of time
    t0_2 = datetime.now()
    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    dt = datetime.now() - t0_2

    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1


    state = next_state
    total_t += 1

    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon



if __name__ == '__main__':

  # hyperparams and initialize stuff
  conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_layer_sizes = [512]
  gamma = 0.99
  batch_sz = 32
  num_episodes = 10000
  total_t = 0
  experience_replay_buffer = []
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
    gamma=gamma,
    # scope="model"
  )
  target_model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    # scope="target_model"
  )



  with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())


    print("Populating experience replay buffer...")
    obs = env.reset()
    obs_small = downsample_image(obs)
    state = np.stack([obs_small] * 4, axis=0)
    # assert(state.shape == (4, 80, 80))
    for i in range(MIN_EXPERIENCES):

        action = np.random.choice(K)
        obs, reward, done, _ = env.step(action)
        next_state = update_state(state, obs)
        # assert(state.shape == (4, 80, 80))
        experience_replay_buffer.append((state, action, reward, next_state, done))

        if done:
            obs = env.reset()
            obs_small = downsample_image(obs)
            state = np.stack([obs_small] * 4, axis=0)
            # assert(state.shape == (4, 80, 80))
        else:
            state = next_state


    # Play a number of episodes and learn!
    for i in range(num_episodes):

      total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
        env,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
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

