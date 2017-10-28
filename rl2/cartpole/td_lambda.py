# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning import FeatureTransformer
from q_learning_bins import plot_running_avg


class SGDRegressor:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)

  def partial_fit(self, x, y, e, lr=1e-1):
    self.w += lr*(y - x.dot(self.w))*e

  def predict(self, X):
    X = np.array(X)
    return X.dot(self.w)


# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer

    sample_feature = feature_transformer.transform( [env.reset()] )
    D = sample_feature.shape[1]

    for i in range(env.action_space.n):
      # model = SGDRegressor(learning_rate="constant")
      # model.partial_fit(feature_transformer.transform( [env.reset()] ), [0])
      model = SGDRegressor(D)
      self.models.append(model)
    
    self.eligibilities = np.zeros((env.action_space.n, D))

  def reset(self):
    self.eligibilities = np.zeros_like(self.eligibilities)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    result = np.stack([m.predict(X) for m in self.models]).T
    return result

  def update(self, s, a, G, gamma, lambda_):
    X = self.feature_transformer.transform([s])
    # assert(len(X.shape) == 2)

    # slower
    # for action in range(self.env.action_space.n):
    #   if action != a:
    #     self.eligibilities[action] *= gamma*lambda_
    #   else:
    #     self.eligibilities[a] = grad + gamma*lambda_*self.eligibilities[a]

    self.eligibilities *= gamma*lambda_
    self.eligibilities[a] += X[0]
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma, lambda_):
  observation = env.reset()
  done = False
  totalreward = 0
  states_actions_rewards = []
  iters = 0
  model.reset()
  while not done and iters < 1000000:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -300

    # update the model
    next = model.predict(observation)
    assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next[0])
    model.update(prev_observation, action, G, gamma, lambda_)

    states_actions_rewards.append((prev_observation, action, reward))

    if reward == 1: # if we changed the reward to -200
      totalreward += reward

    iters += 1

    # if iters > 0 and iters % 1000 == 0:
    #   print(iters)
    # if done:
    #   print "finished in < 1000 steps!"

  return states_actions_rewards, totalreward


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft)
  gamma = 0.999
  lambda_ = 0.7

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 500
  totalrewards = np.empty(N)
  # costs = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    # eps = 0.1*(0.97**n)
    eps = 1.0/np.sqrt(n+1)
    # eps = 0.1
    states_actions_rewards, totalreward = play_one(model, env, eps, gamma, lambda_)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


