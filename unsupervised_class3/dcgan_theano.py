# https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders
# https://www.udemy.com/deep-learning-gans-and-variational-autoencoders
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import util
import scipy as sp
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from datetime import datetime
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test
from theano.tensor.nnet import conv2d


# some constants
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
EPSILON = 1e-8
BATCH_SIZE = 64
EPOCHS = 2
BATCH_SIZE = 64
SAVE_SAMPLE_PERIOD = 50


# make dir to save samples
if not os.path.exists('samples'):
  os.mkdir('samples')


# make it callable with only 1 input
def lrelu(x, alpha=0.2):
  return T.nnet.relu(x, alpha)


# helper for adam optimizer
def adam(params, grads):
  updates = []
  time = theano.shared(0)
  new_time = time + 1
  updates.append((time, new_time))
  lr = LEARNING_RATE*T.sqrt(1 - BETA2**new_time) / (1 - BETA1**new_time)
  for p, g in zip(params, grads):
    m = theano.shared(p.get_value() * 0.)
    v = theano.shared(p.get_value() * 0.)
    new_m = BETA1*m + (1 - BETA1)*g
    new_v = BETA2*v + (1 - BETA2)*g*g
    new_p = p - lr*new_m / (T.sqrt(new_v) + EPSILON)
    updates.append((m, new_m))
    updates.append((v, new_v))
    updates.append((p, new_p))
  return updates


# helper for batch norm
def batch_norm(
  input_,
  gamma,
  beta,
  running_mean,
  running_var,
  is_training,
  axes='per-activation'):

  if is_training:
    # returns:
    #   batch-normalized output
    #   batch mean
    #   batch variance
    #   running mean (for later use as population mean estimate)
    #   running var (for later use as population var estimate)
    out, _, _, running_mean, running_var = batch_normalization_train(
      input_,
      gamma,
      beta,
      running_mean=running_mean,
      running_var=running_var,
      axes=axes,
      running_average_factor=0.9,
    )
  else:
    out = batch_normalization_test(
      input_,
      gamma,
      beta,
      running_mean,
      running_var,
      axes=axes,
    )
  return out, running_mean, running_var


class ConvLayer:
  def __init__(self, mi, mo, apply_batch_norm, filtersz=5, stride=2, f=T.nnet.relu):
    # mi = input feature map size
    # mo = output feature map size
    W = 0.02*np.random.randn(mo, mi, filtersz, filtersz)
    self.W = theano.shared(W)
    self.b = theano.shared(np.zeros(mo))
    self.params = [self.W, self.b]

    if apply_batch_norm:
      self.gamma = theano.shared(np.ones(mo))
      self.beta = theano.shared(np.zeros(mo))
      self.params += [self.gamma, self.beta]

      self.running_mean = T.zeros(mo)
      self.running_var = T.zeros(mo)

    self.f = f
    self.stride = stride
    self.apply_batch_norm = apply_batch_norm
    

  def forward(self, X, is_training):
    conv_out = conv2d(
      input=X,
      filters=self.W,
      subsample=(self.stride, self.stride),
      border_mode='half',
    )
    conv_out += self.b.dimshuffle('x', 0, 'x', 'x')

    # apply batch normalization
    if self.apply_batch_norm:
      conv_out, self.running_mean, self.running_var = batch_norm(
        conv_out,
        self.gamma,
        self.beta,
        self.running_mean,
        self.running_var,
        is_training,
        'spatial'
      )
    return self.f(conv_out)


# regular convolution expects output size to be:
# new_dim = floor( (old_dim - filter_sz) / stride ) + 1

# therefore, for fs-conv, output size should be:
# new_dim = stride * (old_dim - 1) + filter_sz


class FractionallyStridedConvLayer:
  def __init__(self, mi, mo, output_shape, apply_batch_norm, filtersz=5, stride=2, f=T.nnet.relu):
    # mi = input feature map size
    # mo = output feature map size
    self.filter_shape = (mi, mo, filtersz, filtersz)
    W = 0.02*np.random.randn(*self.filter_shape)
    self.W = theano.shared(W)
    self.b = theano.shared(np.zeros(mo))
    self.params = [self.W, self.b]

    if apply_batch_norm:
      self.gamma = theano.shared(np.ones(mo))
      self.beta = theano.shared(np.zeros(mo))
      self.params += [self.gamma, self.beta]

      self.running_mean = T.zeros(mo)
      self.running_var = T.zeros(mo)

    self.f = f
    self.stride = stride
    self.output_shape = output_shape
    self.apply_batch_norm = apply_batch_norm
    self.params = [self.W, self.b]

  def forward(self, X, is_training):
    conv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
      X, self.W,
      input_shape=self.output_shape,
      filter_shape=self.filter_shape,
      border_mode='half',
      subsample=(self.stride, self.stride)
    )
    conv_out += self.b.dimshuffle('x', 0, 'x', 'x')

    # apply batch normalization
    if self.apply_batch_norm:
      conv_out, self.running_mean, self.running_var = batch_norm(
        conv_out,
        self.gamma,
        self.beta,
        self.running_mean,
        self.running_var,
        is_training,
        'spatial'
      )

    return self.f(conv_out)


class DenseLayer(object):
  def __init__(self, M1, M2, apply_batch_norm, f=T.nnet.relu):
    W = 0.02*np.random.randn(M1, M2)
    self.W = theano.shared(W)
    self.b = theano.shared(np.zeros(M2))
    self.params = [self.W, self.b]

    if apply_batch_norm:
      self.gamma = theano.shared(np.ones(M2))
      self.beta = theano.shared(np.zeros(M2))
      self.params += [self.gamma, self.beta]

      self.running_mean = T.zeros(M2)
      self.running_var = T.zeros(M2)

    self.f = f
    self.apply_batch_norm = apply_batch_norm
    self.params = [self.W, self.b]

  def forward(self, X, is_training):
    a = X.dot(self.W) + self.b

    # apply batch normalization
    if self.apply_batch_norm:
      a, self.running_mean, self.running_var = batch_norm(
        a,
        self.gamma,
        self.beta,
        self.running_mean,
        self.running_var,
        is_training,
        'spatial'
      )
    return self.f(a)


class DCGAN:
  def __init__(self, img_length, num_colors, d_sizes, g_sizes):

    # save for later
    self.img_length = img_length
    self.num_colors = num_colors
    self.latent_dims = g_sizes['z']

    # define the input data
    self.X = T.tensor4('placeholderX')
    self.Z = T.matrix('placeholderZ')

    # build the discriminator
    p_real_given_real = self.build_discriminator(self.X, d_sizes)

    # build generator
    self.sample_images = self.build_generator(self.Z, g_sizes)

    # get sample predictions
    p_real_given_fake = self.d_forward(self.sample_images, True)

    # sample with batch norm in test mode
    self.sample_images_test = self.g_forward(self.Z, False)

    # build costs
    self.d_cost_real = T.nnet.binary_crossentropy(
      output=p_real_given_real,
      target=T.ones_like(p_real_given_real),
    )
    self.d_cost_fake = T.nnet.binary_crossentropy(
      output=p_real_given_fake,
      target=T.zeros_like(p_real_given_fake),
    )
    self.d_cost = T.mean(self.d_cost_real) + T.mean(self.d_cost_fake)

    self.g_cost = T.mean(
      T.nnet.binary_crossentropy(
        output=p_real_given_fake,
        target=T.ones_like(p_real_given_fake),
      )
    )
    real_predictions = T.sum(T.eq(T.round(p_real_given_real), 1))
    fake_predictions = T.sum(T.eq(T.round(p_real_given_fake), 0))
    num_predictions = 2.0*BATCH_SIZE
    num_correct = real_predictions + fake_predictions
    self.d_accuracy = num_correct / num_predictions


    # optimizers
    d_grads = T.grad(self.d_cost, self.d_params)
    d_updates = adam(self.d_params, d_grads)
    self.train_d = theano.function(
      inputs=[self.X, self.Z],
      outputs=[self.d_cost, self.d_accuracy],
      updates=d_updates,
    )

    g_grads = T.grad(self.g_cost, self.g_params)
    g_updates = adam(self.g_params, g_grads)
    self.train_g = theano.function(
      inputs=[self.X, self.Z],
      outputs=self.g_cost,
      updates=g_updates,
    )

    # make a function to get sample images
    self.get_sample_images = theano.function(
      inputs=[self.Z],
      outputs=self.sample_images_test,
    )


  def build_discriminator(self, X, d_sizes):
    self.d_params = []

    # build conv layers
    self.d_convlayers = []
    mi = self.num_colors
    dim = self.img_length
    print("*** conv layer image sizes:")
    for mo, filtersz, stride, apply_batch_norm in d_sizes['conv_layers']:
      layer = ConvLayer(mi, mo, apply_batch_norm, filtersz, stride, lrelu)
      self.d_convlayers.append(layer)
      self.d_params += layer.params
      mi = mo
      print("dim:", dim)
      dim = int(np.ceil(float(dim) / stride))

      # for 'valid' border mode
      # dim = int(np.floor( (dim - filtersz) / stride ) ) + 1

      # for 'full' border mode
      # dim = int(np.ceil( (dim + filtersz - 1) / stride ) )

    print("final dim before flatten:", dim)


    mi = mi * dim * dim
    # build dense layers
    self.d_denselayers = []
    for mo, apply_batch_norm in d_sizes['dense_layers']:
      layer = DenseLayer(mi, mo, apply_batch_norm, lrelu)
      mi = mo
      self.d_denselayers.append(layer)
      self.d_params += layer.params


    # final logistic layer
    self.d_finallayer = DenseLayer(mi, 1, False, T.nnet.sigmoid)
    self.d_params += self.d_finallayer.params

    # get the logits
    p_real_given_x = self.d_forward(X, True)

    # build the cost later
    return p_real_given_x


  def d_forward(self, X, is_training):
    # encapsulate this because we use it twice
    output = X
    for layer in self.d_convlayers:
      output = layer.forward(output, is_training)
    output = output.flatten(ndim=2)
    for layer in self.d_denselayers:
      output = layer.forward(output, is_training)
    output = self.d_finallayer.forward(output, is_training)
    return output


  def build_generator(self, Z, g_sizes):
    self.g_params = []

    # determine the size of the data at each step
    dims = [self.img_length]
    dim = self.img_length
    for _, filtersz, stride, _ in reversed(g_sizes['conv_layers']):
      dim = int(np.ceil(float(dim) / stride))

      # for 'valid' border mode
      # dim = int(np.floor( (dim - filtersz) / stride ) ) + 1

      # for 'full' border mode
      # dim = int(np.ceil( (dim + filtersz - 1) / stride ) )

      dims.append(dim)

    # note: dims is actually backwards
    # the first layer of the generator is actually last
    # so let's reverse it
    dims = list(reversed(dims))
    print("dims:", dims)
    self.g_dims = dims


    # dense layers
    mi = self.latent_dims
    self.g_denselayers = []
    for mo, apply_batch_norm in g_sizes['dense_layers']:
      layer = DenseLayer(mi, mo, apply_batch_norm)
      self.g_denselayers.append(layer)
      self.g_params += layer.params
      mi = mo

    # final dense layer
    mo = g_sizes['projection'] * dims[0] * dims[0]
    layer = DenseLayer(mi, mo, not g_sizes['bn_after_project'])
    self.g_denselayers.append(layer)
    self.g_params += layer.params


    # fs-conv layers
    mi = g_sizes['projection']
    self.g_convlayers = []

    # output may use tanh or sigmoid
    num_relus = len(g_sizes['conv_layers']) - 1
    activation_functions = [T.nnet.relu]*num_relus + [g_sizes['output_activation']]

    for i in range(len(g_sizes['conv_layers'])):
      mo, filtersz, stride, apply_batch_norm = g_sizes['conv_layers'][i]
      f = activation_functions[i]
      output_shape = [BATCH_SIZE, mo, dims[i+1], dims[i+1]]
      print("mi:", mi, "mo:", mo, "outp shape:", output_shape)
      layer = FractionallyStridedConvLayer(
        mi, mo, output_shape, apply_batch_norm, filtersz, stride, f
      )
      self.g_convlayers.append(layer)
      self.g_params += layer.params
      mi = mo

    # apply batch norm
    if g_sizes['bn_after_project']:
      self.gamma = theano.shared(np.ones(g_sizes['projection']))
      self.beta = theano.shared(np.zeros(g_sizes['projection']))
      self.running_mean = T.zeros(g_sizes['projection'])
      self.running_var = T.zeros(g_sizes['projection'])
      self.g_params += [self.gamma, self.beta]

    # get the output
    self.g_sizes = g_sizes
    return self.g_forward(Z, True)


  def g_forward(self, Z, is_training):
    # dense layers
    output = Z
    for layer in self.g_denselayers:
      output = layer.forward(output, is_training)

    # project and reshape
    # remember! (N, color, D, D)
    output = output.reshape(
      [-1, self.g_sizes['projection'], self.g_dims[0], self.g_dims[0]],
    )

    # apply batch norm
    if self.g_sizes['bn_after_project']:
      output, self.running_mean, self.running_var = batch_norm(
        output,
        self.gamma,
        self.beta,
        self.running_mean,
        self.running_var,
        is_training,
        'spatial'
      )

    # pass through fs-conv layers
    for layer in self.g_convlayers:
      output = layer.forward(output, is_training)

    return output


  def fit(self, X):
    d_costs = []
    g_costs = []

    N = len(X)
    n_batches = N // BATCH_SIZE
    total_iters = 0
    for i in range(EPOCHS):
      print("epoch:", i)
      np.random.shuffle(X)
      for j in range(n_batches):
        t0 = datetime.now()

        if type(X[0]) is str:
          # is celeb dataset
          batch = util.files2images_theano(
            X[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
          )

        else:
          # is mnist dataset
          batch = X[j*BATCH_SIZE:(j+1)*BATCH_SIZE]

        Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, self.latent_dims))

        # train the discriminator
        d_cost, d_acc = self.train_d(batch, Z)
        d_costs.append(d_cost)

        # train the generator
        g_cost1 = self.train_g(batch, Z)
        g_cost2 = self.train_g(batch, Z)
        g_costs.append((g_cost1 + g_cost2)/2) # just use the avg

        print("  batch: %d/%d - dt: %s - d_acc: %.2f" % (j+1, n_batches, datetime.now() - t0, d_acc))

        # save samples periodically
        total_iters += 1
        if total_iters % SAVE_SAMPLE_PERIOD == 0:
          print("saving a sample...")
          samples = self.sample(64) # shape is (64, D, D, color)

          # for convenience
          d = self.img_length
          
          if samples.shape[-1] == 1:
            # if color == 1, we want a 2-D image (N x N)
            samples = samples.reshape(64, d, d)
            flat_image = np.empty((8*d, 8*d))

            k = 0
            for i in range(8):
              for j in range(8):
                flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k].reshape(d, d)
                k += 1

            # plt.imshow(flat_image, cmap='gray')
          else:
            # if color == 3, we want a 3-D image (N x N x 3)
            flat_image = np.empty((8*d, 8*d, 3))
            k = 0
            for i in range(8):
              for j in range(8):
                # note: we have to change it back to (D, D, color)
                flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k].transpose((1, 2, 0))
                k += 1
            # plt.imshow(flat_image)
            
          # plt.savefig('samples/samples_at_iter_%d.png' % total_iters)
          sp.misc.imsave(
            'samples/samples_at_iter_%d.png' % total_iters,
            flat_image,
          )

    # save a plot of the costs
    plt.clf()
    plt.plot(d_costs, label='discriminator cost')
    plt.plot(g_costs, label='generator cost')
    plt.legend()
    plt.savefig('cost_vs_iteration.png')

  def sample(self, n):
    Z = np.random.uniform(-1, 1, size=(n, self.latent_dims))
    return self.get_sample_images(Z)



def celeb():
  X = util.get_celeb()
  # just loads a list of filenames, we will load them in dynamically
  # because there are many
  dim = 64
  colors = 3

  # for celeb
  d_sizes = {
    'conv_layers': [
      (64, 5, 2, False),
      (128, 5, 2, True),
      (256, 5, 2, True),
      (512, 5, 2, True)
    ],
    'dense_layers': [],
  }
  g_sizes = {
    'z': 100,
    'projection': 512,
    'bn_after_project': True,
    'conv_layers': [
      (256, 5, 2, True),
      (128, 5, 2, True),
      (64, 5, 2, True),
      (colors, 5, 2, False)
    ],
    'dense_layers': [],
    'output_activation': T.tanh,
  }

  # setup gan
  # note: assume square images, so only need 1 dim
  gan = DCGAN(dim, colors, d_sizes, g_sizes)
  gan.fit(X)


def mnist():
  X, Y = util.get_mnist()
  X = X.reshape(len(X), 1, 28, 28) # remember! (N, color, D, D)
  dim = X.shape[2]
  colors = X.shape[1]

  # for mnist
  d_sizes = {
    'conv_layers': [(2, 5, 2, False), (64, 5, 2, True)],
    'dense_layers': [(1024, True)],
  }
  g_sizes = {
    'z': 100,
    'projection': 128,
    'bn_after_project': False,
    'conv_layers': [(128, 5, 2, True), (colors, 5, 2, False)],
    'dense_layers': [(1024, True)],
    'output_activation': T.nnet.sigmoid,
  }


  # setup gan
  # note: assume square images, so only need 1 dim
  gan = DCGAN(dim, colors, d_sizes, g_sizes)
  gan.fit(X)

  # since training will take a considerable
  # amount of time, let's just save some
  # samples to disk rather than plotting now


if __name__ == '__main__':
  # celeb()
  mnist()
