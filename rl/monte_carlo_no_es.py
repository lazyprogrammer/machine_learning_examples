# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')



def epsilon_greedy(policy, s, eps=0.1):
  p = np.random.random()
  if p < (1 - eps):
    return policy[s]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid, policy, max_steps=20):

  # start state
  s = grid.reset()

  # choose action
  a = epsilon_greedy(policy, s)

  states = [s]
  actions = [a]
  rewards = [0]

  for _ in range(max_steps):
    r = grid.move(a)
    s = grid.current_state()

    rewards.append(r)
    states.append(s)
    
    if grid.game_over():
      break
    else:
      a = epsilon_greedy(policy, s)
      actions.append(a)

  # we want to return:
  # states  = [s(0), s(1), ..., s(T-1), s(T)]
  # actions = [a(0), a(1), ..., a(T-1),     ]
  # rewards = [   0, R(1), ..., R(T-1), R(T)]

  return states, actions, rewards


def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often

  # find max val
  max_val = max(d.values())

  # find keys corresponding to max val
  max_keys = [key for key, val in d.items() if val == max_val]

  ### slow version
  # max_keys = []
  # for key, val in d.items():
  #   if val == max_val:
  #     max_keys.append(key)

  return np.random.choice(max_keys), max_val


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()
  # try the negative grid too, to see if agent will learn to go past the "bad spot"
  # in order to minimize number of steps
  # grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  # initialize a random policy
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

  # initialize Q(s,a) and returns
  Q = {}
  sample_counts = {}
  state_sample_count = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions: # not a terminal state
      Q[s] = {}
      sample_counts[s] = {}
      state_sample_count[s] = 0
      for a in ALL_POSSIBLE_ACTIONS:
        Q[s][a] = 0
        sample_counts[s][a] = 0
    else:
      # terminal state or state we can't otherwise get to
      pass

  # repeat until convergence
  deltas = []
  for it in range(10000):
    if it % 1000 == 0:
      print(it)

    # generate an episode using pi
    biggest_change = 0
    states, actions, rewards = play_game(grid, policy)

    # create a list of only state-action pairs for lookup
    states_actions = list(zip(states, actions))

    T = len(states)
    G = 0
    for t in range(T - 2, -1, -1):
      # retrieve current s, a, r tuple
      s = states[t]
      a = actions[t]

      # update G
      G = rewards[t+1] + GAMMA * G

      # check if we have already seen (s, a) ("first-visit")
      if (s, a) not in states_actions[:t]:
        old_q = Q[s][a]
        sample_counts[s][a] += 1
        lr = 1 / sample_counts[s][a]
        Q[s][a] = old_q + lr * (G - old_q)

        # update policy
        policy[s] = max_dict(Q[s])[0]

        # update state sample count
        state_sample_count[s] += 1

        # update delta
        biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
    deltas.append(biggest_change)

  plt.plot(deltas)
  plt.show()

  print("final policy:")
  print_policy(policy, grid)

  # find V
  V = {}
  for s, Qs in Q.items():
    V[s] = max_dict(Q[s])[1]

  print("final values:")
  print_values(V, grid)

  print("state_sample_count:")
  state_sample_count_arr = np.zeros((grid.rows, grid.cols))
  for i in range(grid.rows):
    for j in range(grid.cols):
      if (i, j) in state_sample_count:
        state_sample_count_arr[i,j] = state_sample_count[(i, j)]
  df = pd.DataFrame(state_sample_count_arr)
  print(df)
