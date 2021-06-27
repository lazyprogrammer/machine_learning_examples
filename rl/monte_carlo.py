# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

GAMMA = 0.9

# NOTE: this is only policy evaluation, not optimization

def play_game(grid, policy, max_steps=20):
  # returns a list of states and corresponding returns

  # reset game to start at a random position
  # we need to do this, because given our current deterministic policy
  # we would never end up at certain states, but we still want to measure their value
  start_states = list(grid.actions.keys())
  start_idx = np.random.choice(len(start_states))
  grid.set_state(start_states[start_idx])

  s = grid.current_state()

  # keep track of all states and rewards encountered
  states = [s]
  rewards = [0]

  steps = 0
  while not grid.game_over():
    a = policy[s]
    r = grid.move(a)
    next_s = grid.current_state()

    # update states and rewards lists
    states.append(next_s)
    rewards.append(r)

    steps += 1
    if steps >= max_steps:
      break

    # update state
    # note: there is no need to store the final terminal state
    s = next_s

  # we want to return:
  # states  = [s(0), s(1), ..., S(T)]
  # rewards = [R(0), R(1), ..., R(T)]

  return states, rewards


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

  # initialize V(s) and returns
  V = {}
  returns = {} # dictionary of state -> list of returns we've received
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      returns[s] = []
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  # repeat
  for _ in range(100):
    # generate an episode using pi
    states, rewards = play_game(grid, policy)
    G = 0
    T = len(states)
    for t in range(T - 2, -1, -1):
      s = states[t]
      r = rewards[t+1]
      G = r + GAMMA * G # update return

      # we'll use first-visit Monte Carlo
      if s not in states[:t]:
        returns[s].append(G)
        V[s] = np.mean(returns[s])

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
