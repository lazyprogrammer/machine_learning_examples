# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
import gym
# Wiki:
# https://github.com/openai/gym/wiki/CartPole-v0
# Environment page:
# https://gym.openai.com/envs/CartPole-v0

# get the environment
env = gym.make('CartPole-v0')

# put yourself in the start state
# it also returns the state
env.reset()
# Out[50]: array([-0.04533731, -0.03231478, -0.01469216,  0.04151   ])

# what do the state variables mean?
# Num Observation Min Max
# 0 Cart Position -2.4  2.4
# 1 Cart Velocity -Inf  Inf
# 2 Pole Angle  ~ -41.8°  ~ 41.8°
# 3 Pole Velocity At Tip  -Inf  Inf

box = env.observation_space

# In [53]: box
# Out[53]: Box(4,)

# In [54]: box.
# box.contains       box.high           box.sample         box.to_jsonable
# box.from_jsonable  box.low            box.shape

env.action_space

# In [71]: env.action_space
# Out[71]: Discrete(2)

# In [72]: env.action_space.
# env.action_space.contains       env.action_space.n              env.action_space.to_jsonable
# env.action_space.from_jsonable  env.action_space.sample

# pick an action
action = env.action_space.sample()

# do an action
observation, reward, done, info = env.step(action)


# run through an episode
done = False
while not done:
  observation, reward, done, _ = env.step(env.action_space.sample())



