# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

# import multiprocessing
# from multiprocessing.dummy import Pool

# INSTRUCTIONS FOR INSTALLING PLE:
# https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
from ple import PLE
from ple.games.flappybird import FlappyBird

import sys


# thread pool for parallelization
# pool = Pool(4)


HISTORY_LENGTH = 1


class Env:
  def __init__(self):
    self.game = FlappyBird(pipe_gap=125)
    self.env = PLE(self.game, fps=30, display_screen=False)
    self.env.init()
    self.env.getGameState = self.game.getGameState # maybe not necessary

    # by convention we want to use (0,1)
    # but the game uses (None, 119)
    self.action_map = self.env.getActionSet() #[None, 119]

  def step(self, action):
    action = self.action_map[action]
    reward = self.env.act(action)
    done = self.env.game_over()
    obs = self.get_observation()
    # don't bother returning an info dictionary like gym
    return obs, reward, done

  def reset(self):
    self.env.reset_game()
    return self.get_observation()

  def get_observation(self):
    # game state returns a dictionary which describes
    # the meaning of each value
    # we only want the values
    obs = self.env.getGameState()
    return np.array(list(obs.values()))

  def set_display(self, boolean_value):
    self.env.display_screen = boolean_value


# make a global environment to be used throughout the script
env = Env()


### neural network

# hyperparameters
D = len(env.reset())*HISTORY_LENGTH
M = 50
K = 2

def softmax(a):
  c = np.max(a, axis=1, keepdims=True)
  e = np.exp(a - c)
  return e / e.sum(axis=-1, keepdims=True)

def relu(x):
  return x * (x > 0)

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
    return softmax(Z.dot(self.W2) + self.b2)

  def sample_action(self, x):
    # assume input is a single state of size (D,)
    # first make it (N, D) to fit ML conventions
    X = np.atleast_2d(x)
    P = self.forward(X)
    p = P[0] # the first row
    # return np.random.choice(len(p), p=p)
    return np.argmax(p)

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

    ### slow way
    R = np.zeros(population_size) # stores the reward

    # loop through each "offspring"
    for j in range(population_size):
      params_try = params + sigma*N[j]
      R[j] = f(params_try)

    ### fast way
    # R = pool.map(f, [params + sigma*N[j] for j in range(population_size)])
    # R = np.array(R)

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
    lr *= 0.992354
    # sigma *= 0.99

    print("Iter:", t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:", (datetime.now() - t0))

  return params, reward_per_iteration


def reward_function(params):
  model = ANN(D, M, K)
  model.set_params(params)
  
  # play one episode and return the total reward
  episode_reward = 0
  episode_length = 0 # not sure if it will be used
  done = False
  obs = env.reset()
  obs_dim = len(obs)
  if HISTORY_LENGTH > 1:
    state = np.zeros(HISTORY_LENGTH*obs_dim) # current state
    state[-obs_dim:] = obs
  else:
    state = obs
  while not done:
    # get the action
    action = model.sample_action(state)

    # perform the action
    obs, reward, done = env.step(action)

    # update total reward
    episode_reward += reward
    episode_length += 1

    # update state
    if HISTORY_LENGTH > 1:
      state = np.roll(state, -obs_dim)
      state[-obs_dim:] = obs
    else:
      state = obs
  return episode_reward


if __name__ == '__main__':
  model = ANN(D, M, K)

  if len(sys.argv) > 1 and sys.argv[1] == 'play':
    # play with a saved model
    j = np.load('es_flappy_results.npz')
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
      'es_flappy_results.npz',
      train=rewards,
      **model.get_params_dict(),
    )

  # play 5 test episodes
  env.set_display(True)
  for _ in range(5):
    print("Test:", reward_function(best_params))

