import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from datetime import datetime

gym_minor_version = int(gym.__version__.split('.')[1])
if gym_minor_version >= 19:
  exit("Please install OpenAI Gym 0.19.0 or earlier")

if tf.__version__.startswith('2'):
  exit("Please install Tensorflow 1.x")


### avoid crashing on Mac
# doesn't seem to work
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


# simple feedforward neural net
def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
  for h in layer_sizes[:-1]:
    x = tf.layers.dense(x, units=h, activation=hidden_activation)
  return tf.layers.dense(x, units=layer_sizes[-1], activation=output_activation)


# get all variables within a scope
def get_vars(scope):
  return [x for x in tf.global_variables() if scope in x.name]


### Create both the actor and critic networks at once ###
### Q(s, mu(s)) returns the maximum Q for a given state s ###
def CreateNetworks(
    s, a,
    num_actions,
    action_max,
    hidden_sizes=(300,),
    hidden_activation=tf.nn.relu, 
    output_activation=tf.tanh):

  with tf.variable_scope('mu'):
    mu = action_max * ANN(s, list(hidden_sizes)+[num_actions], hidden_activation, output_activation)
  with tf.variable_scope('q1'):
    input_ = tf.concat([s, a], axis=-1) # (state, action)
    q1 = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
  with tf.variable_scope('q2'):
    input_ = tf.concat([s, a], axis=-1) # (state, action)
    q2 = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
  with tf.variable_scope('q1', reuse=True):
    # reuse is True, so it reuses the weights from the previously defined Q network
    input_ = tf.concat([s, mu], axis=-1) # (state, mu(state))
    q1_mu = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
  return mu, q1, q2, q1_mu


### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])


### Implement the TD3 algorithm ###
def td3(
    env_fn,
    ac_kwargs=dict(),
    seed=0,
    save_folder=None,
    num_train_episodes=100,
    test_agent_every=25,
    replay_size=int(1e6),
    gamma=0.99, 
    decay=0.995,
    mu_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000, 
    action_noise=0.1,
    target_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
    max_episode_length=1000):

  tf.set_random_seed(seed)
  np.random.seed(seed)

  env, test_env = env_fn(), env_fn()

  # comment out this line if you don't want to record a video of the agent
  if save_folder is not None:
    test_env = gym.wrappers.Monitor(test_env, save_folder)

  # get size of state space and action space
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]

  # Maximum value of action
  # Assumes both low and high values are the same
  # Assumes all actions have the same bounds
  # May NOT be the case for all environments
  action_max = env.action_space.high[0]

  # Create Tensorflow placeholders (neural network inputs)
  X = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # state
  A = tf.placeholder(dtype=tf.float32, shape=(None, num_actions)) # action
  X2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # next state
  R = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
  D = tf.placeholder(dtype=tf.float32, shape=(None,)) # done

  # Main network outputs
  with tf.variable_scope('main'):
    mu, q1, q2, q1_mu = CreateNetworks(X, A, num_actions, action_max, **ac_kwargs)
  
  # Target networks
  # First, get the output policy given next state X2
  with tf.variable_scope('target'):
    # Note: "A" placeholder is effectively ignored
    #       since mu is only a function of state (X2)
    mu_targ, _, _, _ = CreateNetworks(X2, A, num_actions, action_max, **ac_kwargs)

  # Next, add noise to mu_targ, before passing it through the target Q-networks
  with tf.variable_scope('target', reuse=True):
    # Add Gaussian noise and clip to valid action range
    epsilon = tf.random_normal(tf.shape(mu_targ), stddev=target_noise)
    epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
    A2 = mu_targ + epsilon
    A2 = tf.clip_by_value(A2, -action_max, action_max)

    _, q1_targ, q2_targ, _ = CreateNetworks(X2, A2, num_actions, action_max, **ac_kwargs)

  # Experience replay memory
  replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)


  # Target value for the Q-network loss
  # We use stop_gradient to tell Tensorflow not to differentiate
  # Take the smaller of Q1 and Q2!
  min_q_targ = tf.minimum(q1_targ, q2_targ)
  q_target = tf.stop_gradient(R + gamma * (1 - D) * min_q_targ)

  # TD3 losses
  mu_loss = -tf.reduce_mean(q1_mu)
  q1_loss = tf.reduce_mean((q1 - q_target)**2)
  q2_loss = tf.reduce_mean((q2 - q_target)**2)
  q_loss = q1_loss + q2_loss # minimize simultaneously

  # Train policy and value separately
  mu_optimizer = tf.train.AdamOptimizer(learning_rate=mu_lr)
  q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
  mu_train_op = mu_optimizer.minimize(mu_loss, var_list=get_vars('main/mu'))
  q_train_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

  # Use soft updates to update the target networks
  target_update = tf.group(
    [tf.assign(v_targ, decay*v_targ + (1 - decay)*v_main)
      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ]
  )

  # Copy main network params to target networks
  target_init = tf.group(
    [tf.assign(v_targ, v_main)
      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ]
  )

  # boilerplate (and copy to the target networks!)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(target_init)

  def get_action(s, noise_scale):
    a = sess.run(mu, feed_dict={X: s.reshape(1,-1)})[0]
    a += noise_scale * np.random.randn(num_actions)
    return np.clip(a, -action_max, action_max)

  test_returns = []
  def test_agent(num_episodes=5):
    t0 = datetime.now()
    n_steps = 0
    for j in range(num_episodes):
      s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
      while not (d or (episode_length == max_episode_length)):
        # Take deterministic actions at test time (noise_scale=0)
        test_env.render()
        s, r, d, _ = test_env.step(get_action(s, 0))
        episode_return += r
        episode_length += 1
        n_steps += 1
      print('test return:', episode_return, 'episode_length:', episode_length)
      test_returns.append(episode_return)
    # print("test steps per sec:", n_steps / (datetime.now() - t0).total_seconds())


  # Main loop: play episode and train
  returns = []
  q_losses = []
  mu_losses = []
  num_steps = 0
  for i_episode in range(num_train_episodes):

    # reset env
    s, episode_return, episode_length, d = env.reset(), 0, 0, False

    while not (d or (episode_length == max_episode_length)):
      # For the first `start_steps` steps, use randomly sampled actions
      # in order to encourage exploration.
      if num_steps > start_steps:
        a = get_action(s, action_noise)
      else:
        a = env.action_space.sample()

      # Keep track of the number of steps done
      num_steps += 1
      if num_steps == start_steps:
        print("USING AGENT ACTIONS NOW")

      # Step the env
      s2, r, d, _ = env.step(a)
      episode_return += r
      episode_length += 1

      # Ignore the "done" signal if it comes from hitting the time
      # horizon (that is, when it's an artificial terminal signal
      # that isn't based on the agent's state)
      d_store = False if episode_length == max_episode_length else d

      # Store experience to replay buffer
      replay_buffer.store(s, a, r, s2, d_store)

      # Assign next state to be the current state on the next round
      s = s2

    # Perform the updates
    for j in range(episode_length):
      batch = replay_buffer.sample_batch(batch_size)
      feed_dict = {
        X: batch['s'],
        X2: batch['s2'],
        A: batch['a'],
        R: batch['r'],
        D: batch['d']
      }

      # Q network update
      # Note: plot the Q loss if you want
      ql, _ = sess.run([q_loss, q_train_op], feed_dict)
      q_losses.append(ql)

      # Policy update
      # (And target networks update)
      # Note: plot the mu loss if you want
      if j % policy_delay == 0:
        mul, _, _ = sess.run([mu_loss, mu_train_op, target_update], feed_dict)
        mu_losses.append(mul)

    print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length)
    returns.append(episode_return)

    # Test the agent
    if i_episode > 0 and i_episode % test_agent_every == 0:
      test_agent()

  # on Mac, plotting results in an error, so just save the results for later
  # if you're not on Mac, feel free to uncomment the below lines
  np.savez('td3_results.npz', train=returns, test=test_returns, q_losses=q_losses, mu_losses=mu_losses)

  # plt.plot(returns)
  # plt.plot(smooth(np.array(returns)))
  # plt.title("Train returns")
  # plt.show()

  # plt.plot(test_returns)
  # plt.plot(smooth(np.array(test_returns)))
  # plt.title("Test returns")
  # plt.show()

  # plt.plot(q_losses)
  # plt.title('q_losses')
  # plt.show()

  # plt.plot(mu_losses)
  # plt.title('mu_losses')
  # plt.show()


def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
  parser.add_argument('--env', type=str, default='Pendulum-v0')
  parser.add_argument('--hidden_layer_sizes', type=int, default=300)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_train_episodes', type=int, default=200)
  parser.add_argument('--save_folder', type=str, default='td3_monitor')
  args = parser.parse_args()


  td3(
    lambda : gym.make(args.env),
    ac_kwargs=dict(hidden_sizes=[args.hidden_layer_sizes]*args.num_layers),
    gamma=args.gamma,
    seed=args.seed,
    save_folder=args.save_folder,
    num_train_episodes=args.num_train_episodes,
  )
