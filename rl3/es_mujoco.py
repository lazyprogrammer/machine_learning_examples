# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import multiprocessing
from multiprocessing.dummy import Pool

import gym
import sys


# environment
ENV_NAME = 'HalfCheetah-v2'


# thread pool for parallelization
pool = Pool(4)


### neural network

# hyperparameters
env = gym.make(ENV_NAME)
D = len(env.reset())
M = 300
K = env.action_space.shape[0]
action_max = env.action_space.high[0]


def relu(x):
  return x * (x > 0)

# def output_activation(x):
#   return action_max * np.tanh(x)

class ANN:
  def __init__(self, D, M, K, f=relu):
    self.D = D
    self.M = M
    self.K = K
    self.f = f

  def init(self):
    D, M, K = self.D, self.M, self.K
    self.W1 = np.random.randn(D, M) / np.sqrt(D)
    # self.W1 = np.zeros((D, M))
    self.b1 = np.zeros(M)
    self.W2 = np.random.randn(M, K) / np.sqrt(M)
    # self.W2 = np.zeros((M, K))
    self.b2 = np.zeros(K)

  def forward(self, X):
    Z = self.f(X.dot(self.W1) + self.b1)
    return np.tanh(Z.dot(self.W2) + self.b2) * action_max

  def sample_action(self, x):
    # assume input is a single state of size (D,)
    # first make it (N, D) to fit ML conventions
    X = np.atleast_2d(x)
    Y = self.forward(X)
    return Y[0] # the first row

  def get_params(self):
    # return a flat array of parameters
    return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

  def get_params_dict(self):
    return {
      'W1': self.W1,
      'b1': self.b1,
      'W2': self.W2,
      'b2': self.b2,
    }

  def set_params(self, params):
    # params is a flat list
    # unflatten into individual weights
    D, M, K = self.D, self.M, self.K
    self.W1 = params[:D * M].reshape(D, M)
    self.b1 = params[D * M:D * M + M]
    self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
    self.b2 = params[-K:]


def evolution_strategy(
    f,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters):

  # assume initial params is a 1-D array
  num_params = len(initial_params)
  reward_per_iteration = np.zeros(num_iters)

  params = initial_params
  for t in range(num_iters):
    t0 = datetime.now()
    N = np.random.randn(population_size, num_params)

    # ### slow way
    # R = np.zeros(population_size) # stores the reward

    # # loop through each "offspring"
    # for j in range(population_size):
    #   params_try = params + sigma*N[j]
    #   R[j] = f(params_try)

    ### fast way
    R = pool.map(f, [params + sigma*N[j] for j in range(population_size)])
    R = np.array(R)

    m = R.mean()
    s = R.std()
    if s == 0:
      # we can't apply the following equation
      print("Skipping")
      continue

    A = (R - m) / s
    reward_per_iteration[t] = m
    params = params + lr/(population_size*sigma) * np.dot(N.T, A)

    # update the learning rate
    # lr *= 0.992354
    # sigma *= 0.99

    print("Iter:", t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:", (datetime.now() - t0))

  return params, reward_per_iteration


def reward_function(params, display=False):
  model = ANN(D, M, K)
  model.set_params(params)

  env = gym.make(ENV_NAME)
  if display:
    env = gym.wrappers.Monitor(env, 'es_monitor')
  
  # play one episode and return the total reward
  episode_reward = 0
  episode_length = 0 # not sure if it will be used
  done = False
  state = env.reset()
  while not done:
    # display the env
    if display:
      env.render()

    # get the action
    action = model.sample_action(state)

    # perform the action
    state, reward, done, _ = env.step(action)

    # update total reward
    episode_reward += reward
    episode_length += 1

  return episode_reward


if __name__ == '__main__':
  model = ANN(D, M, K)

  if len(sys.argv) > 1 and sys.argv[1] == 'play':
    # play with a saved model
    j = np.load('es_mujoco_results.npz')
    best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])

    # in case initial shapes are not correct
    D, M = j['W1'].shape
    K = len(j['b2'])
    model.D, model.M, model.K = D, M, K
  else:
    # train and save
    model.init()
    params = model.get_params()
    best_params, rewards = evolution_strategy(
      f=reward_function,
      population_size=30,
      sigma=0.1,
      lr=0.03,
      initial_params=params,
      num_iters=300,
    )

    # plot the rewards per iteration
    # plt.plot(rewards)
    # plt.show()
    model.set_params(best_params)
    np.savez(
      'es_mujoco_results.npz',
      train=rewards,
      **model.get_params_dict(),
    )

  # play test episode
  print("Test:", reward_function(best_params, display=True))

