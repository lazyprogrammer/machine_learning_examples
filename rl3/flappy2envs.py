# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from ple import PLE
from ple.games.flappybird import FlappyBird

import sys

from threading import Thread



HISTORY_LENGTH = 1


class Env:
  def __init__(self):
    self.game = FlappyBird(pipe_gap=125)
    self.env = PLE(self.game, fps=30, display_screen=True)
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

  def score(self, X, Y):
    P = np.argmax(self.forward(X), axis=1)
    return np.mean(Y == P)

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



env1, env2 = Env(), Env()




def reward_function(params, env):
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
    state[obs_dim:] = obs
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
  print("Reward:", episode_reward)


if __name__ == '__main__':

  j = np.load('es_flappy_results.npz')
  best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])

  # in case D isn't correct
  D, M = j['W1'].shape
  K = len(j['b2'])
  
  t1 = Thread(target=reward_function, args=(best_params, env1))
  t2 = Thread(target=reward_function, args=(best_params, env2))
  t1.start()
  t2.start()
  t1.join()
  t2.join()


