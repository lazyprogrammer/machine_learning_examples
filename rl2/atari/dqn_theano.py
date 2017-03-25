# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import gym
import os
import sys
import random
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize

if '../cartpole' not in sys.path:
  sys.path.append('../cartpole')
from q_learning_bins import plot_running_avg

# constants
IM_WIDTH = 80
IM_HEIGHT = 80


def init_filter(shape):
  w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]))
  return w.astype(np.float32)


class ConvLayer(object):
  def __init__(self, mi, mo, filtsz=5, stride=2, f=T.nnet.relu):
    # mi = input feature map size
    # mo = output feature map size
    sz = (mo, mi, filtsz, filtsz)
    W0 = init_filter(sz)
    self.W = theano.shared(W0)
    b0 = np.zeros(mo, dtype=np.float32)
    self.b = theano.shared(b0)
    self.stride = (stride, stride)
    self.params = [self.W, self.b]
    self.f = f

  def forward(self, X):
    conv_out = conv2d(
      input=X,
      filters=self.W,
      subsample=self.stride,
    )
    return self.f(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))


def downsample_image(A):
  B = A[31:195] # select the important parts of the image
  B = B / 255.0 # scale to 0..1
  B = B.mean(axis=2) # convert to grayscale

  # downsample image
  # changing aspect ratio doesn't significantly distort the image
  # nearest neighbor interpolation produces a much sharper image
  # than default bilinear
  B = imresize(B, size=(IM_HEIGHT, IM_WIDTH), interp='nearest')
  return B.astype(np.float32)


# a version of HiddenLayer that keeps track of params
class HiddenLayer:
  def __init__(self, M1, M2, f=T.tanh, use_bias=True):
    W = np.random.randn(M1, M2) / np.sqrt(M1+M2)
    self.W = theano.shared(W.astype(np.float32))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = theano.shared(np.zeros(M2).astype(np.float32))
      self.params += [self.b]
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = X.dot(self.W) + self.b
    else:
      a = X.dot(self.W)
    return self.f(a)


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, max_experiences=500000, min_experiences=50000, batch_sz=32):
    self.K = K
    lr = np.float32(2.5e-4)
    mu = np.float32(0)
    decay = np.float32(0.99)

    # inputs and targets
    X = T.ftensor4('X')
    G = T.fvector('G')
    actions = T.ivector('actions')

    # create the graph
    self.conv_layers = []
    num_input_filters = 4 # number of filters / color channels
    for num_output_filters, filtersz, stride in conv_layer_sizes:
      layer = ConvLayer(num_input_filters, num_output_filters, filtersz, stride)
      self.conv_layers.append(layer)
      num_input_filters = num_output_filters

    # get conv output size
    Z = X
    for layer in self.conv_layers:
      Z = layer.forward(Z)
    conv_out = Z.flatten(ndim=2)
    conv_out_op = theano.function(inputs=[X], outputs=conv_out, allow_input_downcast=True)
    test = conv_out_op(np.random.randn(1, 4, IM_HEIGHT, IM_WIDTH))
    flattened_ouput_size = test.shape[1]

    # print("test.shape:", test.shape)
    # print("flattened_ouput_size:", flattened_ouput_size)

    # build fully connected layers
    self.layers = []
    M1 = flattened_ouput_size
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, lambda x: x)
    self.layers.append(layer)

    # collect params for copy
    self.params = []
    for layer in (self.conv_layers + self.layers):
      self.params += layer.params
    caches = [theano.shared(np.ones_like(p.get_value())*0.1) for p in self.params]
    velocities = [theano.shared(p.get_value()*0) for p in self.params]

    # calculate final output and cost
    Z = conv_out
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = Z

    selected_action_values = Y_hat[T.arange(actions.shape[0]), actions]
    cost = T.sum((G - selected_action_values)**2) 

    # create train function
    grads = T.grad(cost, self.params)
    g_update = [(p, p + v) for p, v, g in zip(self.params, velocities, grads)]
    c_update = [(c, decay*c + (np.float32(1) - decay)*g*g) for c, g in zip(caches, grads)]
    v_update = [(v, mu*v - lr*g / T.sqrt(c)) for v, c, g in zip(velocities, caches, grads)]
    # v_update = [(v, mu*v - lr*g) for v, g in zip(velocities, grads)]
    # c_update = []
    updates = c_update + g_update + v_update

    # compile functions
    self.train_op = theano.function(
      inputs=[X, G, actions],
      updates=updates,
      allow_input_downcast=True
    )
    self.predict_op = theano.function(
      inputs=[X],
      outputs=Y_hat,
      allow_input_downcast=True
    )

    # create replay memory
    self.experience = []
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma

  def copy_from(self, other):
    # collect all the ops
    ops = []
    my_params = self.params
    other_params = other.params
    for p, q in zip(my_params, other_params):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)
    # now run them all
    self.session.run(ops)

  def predict(self, X):
    return self.predict_op(X)

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if len(self.experience) < self.min_experiences:
      # don't do anything if we don't have enough experience
      return

    # randomly select a batch
    sample = random.sample(self.experience, self.batch_sz)
    states, actions, rewards, next_states = map(np.array, zip(*sample))
    next_Q = np.max(target_network.predict(next_states), axis=1)
    targets = [r + self.gamma*next_q for r, next_q in zip(rewards, next_Q)]

    # call optimizer
    self.train_op(states, targets, actions)

  def add_experience(self, s, a, r, s2):
    if len(self.experience) >= self.max_experiences:
      self.experience.pop(0)
    if len(s) != 4 or len(s2) != 4:
      print("BAD STATE")
    self.experience.append((s, a, r, s2))

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])


def update_state(state, observation):
  # downsample and grayscale observation
  observation_small = downsample_image(observation)
  state.append(observation_small)
  if len(state) > 4:
    state.pop(0)


def play_one(env, model, tmodel, eps, eps_step, gamma, copy_period):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  state = []
  prev_state = []
  update_state(state, observation) # add the first observation
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early

    if len(state) < 4:
      # we can't choose an action based on model
      action = env.action_space.sample()
    else:
      action = model.sample_action(state, eps)

    # copy state to prev state
    prev_state.append(state[-1])
    if len(prev_state) > 4:
      prev_state.pop(0)

    # perform the action
    observation, reward, done, info = env.step(action)

    # add the new frame to the state
    update_state(state, observation)

    totalreward += reward
    if done:
      reward = -200

    # update the model
    if len(state) == 4 and len(prev_state) == 4:
      model.add_experience(prev_state, action, reward, state)
      model.train(tmodel)

    iters += 1
    eps = max(eps - eps_step, 0.1)

    if iters % copy_period == 0:
      tmodel.copy_from(model)

  return totalreward, eps, iters


def main():
  env = gym.make('Breakout-v0')
  gamma = 0.99
  copy_period = 10000

  D = len(env.observation_space.sample())
  K = env.action_space.n
  conv_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_sizes = [512]
  model = DQN(K, conv_sizes, hidden_sizes, gamma)
  tmodel = DQN(K, conv_sizes, hidden_sizes, gamma)


  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 100000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  n_max = 500000 # last step to decrease epsilon
  eps_step = 0.9 / n_max
  eps = 1.0
  for n in range(N):
    t0 = datetime.now()
    totalreward, eps, num_steps = play_one(env, model, tmodel, eps, eps_step, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", "%.3f" % eps, "num steps:", num_steps, "episode duration:", (datetime.now() - t0), "avg reward (last 100):", "%.3f" % totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()


