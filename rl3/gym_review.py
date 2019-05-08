# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
  return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params):
  observation = env.reset()
  done = False
  t = 0
  r = 0

  while not done and t < 10000:
    t += 1
    action = get_action(observation, params)
    observation, reward, done, info = env.step(action)
    r += reward

  return r


def play_multiple_episodes(env, T, params):
  episode_rewards = np.empty(T)

  for i in range(T):
    episode_rewards[i] = play_one_episode(env, params)

  avg_reward = episode_rewards.mean()
  print("avg reward:", avg_reward)
  return avg_reward


def random_search(env):
  episode_rewards = []
  best = 0
  params = None
  for t in range(100):
    new_params = np.random.random(4)*2 - 1
    avg_reward = play_multiple_episodes(env, 100, new_params)
    episode_rewards.append(avg_reward)

    if avg_reward > best:
      params = new_params
      best = avg_reward
  return episode_rewards, params


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  episode_rewards, params = random_search(env)
  plt.plot(episode_rewards)
  plt.show()

  # play a final set of episodes
  print("***Final run with final weights***")
  play_multiple_episodes(env, 100, params)
