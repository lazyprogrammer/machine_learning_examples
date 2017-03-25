# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

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

if '../cartpole' not in sys.path:
  sys.path.append('../cartpole')
from q_learning_bins import plot_running_avg

# constants
IM_WIDTH = 80
IM_HEIGHT = 80


def downsample_image(A):
  B = A[31:195] # select the important parts of the image
  B = B.mean(axis=2) # convert to grayscale
  B = B / 255.0 # scale to 0..1

  # downsample image
  # changing aspect ratio doesn't significantly distort the image
  # nearest neighbor interpolation produces a much sharper image
  # than default bilinear
  B = imresize(B, size=(IM_HEIGHT, IM_WIDTH), interp='nearest')
  return B


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope, max_experiences=500000, min_experiences=50000, batch_sz=32):
    self.K = K
    self.scope = scope

    with tf.variable_scope(scope):

      # inputs and targets
      self.X = tf.placeholder(tf.float32, shape=(None, 4, IM_HEIGHT, IM_WIDTH), name='X')
      # tensorflow convolution needs the order to be:
      # (num_samples, height, width, "color")
      # so we need to tranpose later
      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

      # calculate output and cost
      # convolutional layers
      # these built-in layers are faster and don't require us to
      # calculate the size of the output of the final conv layer!
      Z = self.X
      Z = tf.transpose(Z, [0, 2, 3, 1])
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

      cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
      # self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
      # self.train_op = tf.train.AdagradOptimizer(10e-3).minimize(cost)
      self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=10e-3).minimize(cost)
      # self.train_op = tf.train.MomentumOptimizer(10e-4, momentum=0.9).minimize(cost)
      # self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)

    # create replay memory
    self.experience = []
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma

  def copy_from(self, other):
    mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      actual = session.run(q)
      op = p.assign(actual)
      ops.append(op)

    self.session.run(ops)

  def set_session(self, session):
    self.session = session

  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if len(self.experience) < self.min_experiences:
      # don't do anything if we don't have enough experience
      return

    # randomly select a batch
    sample = random.sample(self.experience, self.batch_sz)
    states, actions, rewards, next_states = map(np.array, zip(*sample))
    next_Q = np.max(target_network.predict(next_states), axis=1)
    targets = [r + self.gamma*next_q for r, next_q in zip(rewards, next_Q)]

    # call optimizer
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )

  def add_experience(self, s, a, r, s2):
    if len(self.experience) >= self.max_experiences:
      self.experience.pop(0)
    if len(s) != 4 or len(s2) != 4:
      print("BAD STATE")
    self.experience.append((s, a, r, s2))

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])


def update_state(state, observation):
  # downsample and grayscale observation
  observation_small = downsample_image(observation)
  state.append(observation_small)
  if len(state) > 4:
    state.pop(0)


def play_one(env, model, tmodel, eps, eps_step, gamma, copy_period):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  state = []
  prev_state = []
  update_state(state, observation) # add the first observation
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early

    if len(state) < 4:
      # we can't choose an action based on model
      action = env.action_space.sample()
    else:
      action = model.sample_action(state, eps)

    # copy state to prev state
    prev_state.append(state[-1])
    if len(prev_state) > 4:
      prev_state.pop(0)

    # perform the action
    observation, reward, done, info = env.step(action)

    # add the new frame to the state
    update_state(state, observation)

    totalreward += reward
    if done:
      reward = -200

    # update the model
    if len(state) == 4 and len(prev_state) == 4:
      model.add_experience(prev_state, action, reward, state)
      model.train(tmodel)

    iters += 1
    eps = max(eps - eps_step, 0.1)

    if iters % copy_period == 0:
      tmodel.copy_from(model)

  return totalreward, eps, iters


def main():
  env = gym.make('Breakout-v0')
  gamma = 0.99
  copy_period = 10000

  D = len(env.observation_space.sample())
  K = env.action_space.n
  conv_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_sizes = [512]
  model = DQN(K, conv_sizes, hidden_sizes, gamma, scope='main')
  tmodel = DQN(K, conv_sizes, hidden_sizes, gamma, scope='target')
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  model.set_session(session)
  tmodel.set_session(session)


  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 100000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  n_max = 500000 # last step to decrease epsilon
  eps_step = 0.9 / n_max
  eps = 1.0
  for n in range(N):
    t0 = datetime.now()
    totalreward, eps, num_steps = play_one(env, model, tmodel, eps, eps_step, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", "%.3f" % eps, "num steps:", num_steps, "episode duration:", (datetime.now() - t0), "avg reward (last 100):", "%.3f" % totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()


