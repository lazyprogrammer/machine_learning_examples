# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
# Inspired by https://github.com/dennybritz/reinforcement-learning

import gym
import os
import sys
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning_bins import plot_running_avg


# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=T.tanh, use_bias=True):
    self.W = theano.shared(np.random.randn(M1, M2) * np.sqrt(2 / M1))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = theano.shared(np.zeros(M2))
      self.params += [self.b]
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = X.dot(self.W) + self.b
    else:
      a = X.dot(self.W)
    return self.f(a)


# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # learning rate and other hyperparams
    lr = 1e-4

    # create the graph
    # K = number of actions
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
    self.layers.append(layer)

    # get all params for gradient later
    params = []
    for layer in self.layers:
      params += layer.params

    # inputs and targets
    X = T.matrix('X')
    actions = T.ivector('actions')
    advantages = T.vector('advantages')

    # calculate output and cost
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    action_scores = Z
    p_a_given_s = T.nnet.softmax(action_scores)

    selected_probs = T.log(p_a_given_s[T.arange(actions.shape[0]), actions])
    cost = -T.sum(advantages * selected_probs)
    
    # specify update rule
    grads = T.grad(cost, params)
    updates = [(p, p - lr*g) for p, g in zip(params, grads)]

    # compile functions
    self.train_op = theano.function(
      inputs=[X, actions, advantages],
      updates=updates,
      allow_input_downcast=True
    )
    self.predict_op = theano.function(
      inputs=[X],
      outputs=p_a_given_s,
      allow_input_downcast=True
    )

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.train_op(X, actions, advantages)

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.predict_op(X)

  def sample_action(self, X):
    p = self.predict(X)[0]
    nonans = np.all(~np.isnan(p))
    assert(nonans)
    return np.random.choice(len(p), p=p)


# approximates V(s)
class ValueModel:
  def __init__(self, D, hidden_layer_sizes):
    # constant learning rate is fine
    lr = 1e-4

    # create the graph
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x)
    self.layers.append(layer)

    # get all params for gradient later
    params = []
    for layer in self.layers:
      params += layer.params

    # inputs and targets
    X = T.matrix('X')
    Y = T.vector('Y')

    # calculate output and cost
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = T.flatten(Z)
    cost = T.sum((Y - Y_hat)**2)

    # specify update rule
    grads = T.grad(cost, params)
    updates = [(p, p - lr*g) for p, g in zip(params, grads)]

    # compile functions
    self.train_op = theano.function(
      inputs=[X, Y],
      updates=updates,
      allow_input_downcast=True
    )
    self.predict_op = theano.function(
      inputs=[X],
      outputs=Y_hat,
      allow_input_downcast=True
    )

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    self.train_op(X, Y)

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.predict_op(X)


# def play_one_td(env, pmodel, vmodel, gamma):
#   observation = env.reset()
#   done = False
#   totalreward = 0
#   iters = 0

#   while not done and iters < 2000:
#     # if we reach 2000, just quit, don't want this going forever
#     # the 200 limit seems a bit early
#     action = pmodel.sample_action(observation)
#     prev_observation = observation
#     observation, reward, done, info = env.step(action)

#     if done:
#       reward = -200

#     # update the models
#     V_next = vmodel.predict(observation)
#     G = reward + gamma*np.max(V_next)
#     advantage = G - vmodel.predict(prev_observation)
#     pmodel.partial_fit(prev_observation, action, advantage)
#     vmodel.partial_fit(prev_observation, G)

#     if reward == 1: # if we changed the reward to -200
#       totalreward += reward
#     iters += 1

#   return totalreward


def play_one_mc(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  states = []
  actions = []
  rewards = []

  reward = 0
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)

    states.append(observation)
    actions.append(action)
    rewards.append(reward)

    prev_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -200

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  # save the final (s,a,r) tuple
  action = pmodel.sample_action(observation)
  states.append(observation)
  actions.append(action)
  rewards.append(reward)

  returns = []
  advantages = []
  G = 0
  for s, r in zip(reversed(states), reversed(rewards)):
    returns.append(G)
    advantages.append(G - vmodel.predict(s)[0])
    G = r + gamma*G
  returns.reverse()
  advantages.reverse()

  # update the models
  pmodel.partial_fit(states[1:], actions[1:], advantages[1:])
  vmodel.partial_fit(states, returns)

  return totalreward


def main():
  env = gym.make('CartPole-v0')
  D = env.observation_space.shape[0]
  K = env.action_space.n
  pmodel = PolicyModel(D, K, [])
  vmodel = ValueModel(D, [10])
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 1000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    totalreward = play_one_mc(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()

