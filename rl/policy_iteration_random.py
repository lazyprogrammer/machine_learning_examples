# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# next state and reward will now have some randomness
# you'll go in your desired direction with probability 0.5
# you'll go in a random direction a' != a with probability 0.5/3

if __name__ == '__main__':
  # this grid gives you a reward of -0.1 for every non-terminal state
  # we want to see if this will encourage finding a shorter path to the goal
  grid = negative_grid(step_cost=-1.0)
  # grid = negative_grid(step_cost=-0.1)
  # grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  # we'll randomly choose an action and update as we learn
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

  # initial policy
  print("initial policy:")
  print_policy(policy, grid)

  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    # V[s] = 0
    if s in grid.actions:
      V[s] = np.random.random()
    else:
      # terminal state
      V[s] = 0

  # repeat until convergence - will break out when policy does not change
  while True:

    # policy evaluation step - we already know how to do this!
    while True:
      biggest_change = 0
      for s in states:
        old_v = V[s]

        # V(s) only has value if it's not a terminal state
        new_v = 0
        if s in policy:
          for a in ALL_POSSIBLE_ACTIONS:
            if a == policy[s]:
              p = 0.5
            else:
              p = 0.5/3
            grid.set_state(s)
            r = grid.move(a)
            new_v += p*(r + GAMMA * V[grid.current_state()])
          V[s] = new_v
          biggest_change = max(biggest_change, np.abs(old_v - V[s]))

      if biggest_change < SMALL_ENOUGH:
        break

    # policy improvement step
    is_policy_converged = True
    for s in states:
      if s in policy:
        old_a = policy[s]
        new_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in ALL_POSSIBLE_ACTIONS: # chosen action
          v = 0
          for a2 in ALL_POSSIBLE_ACTIONS: # resulting action
            if a == a2:
              p = 0.5
            else:
              p = 0.5/3
            grid.set_state(s)
            r = grid.move(a2)
            v += p*(r + GAMMA * V[grid.current_state()])
          if v > best_value:
            best_value = v
            new_a = a
        policy[s] = new_a
        if new_a != old_a:
          is_policy_converged = False

    if is_policy_converged:
      break

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
  # result: every move is as bad as losing, so lose as quickly as possible
