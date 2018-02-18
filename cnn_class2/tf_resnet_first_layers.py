# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


# Let's go up to the end of the first conv block
# to make sure everything has been loaded correctly
# compared to keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock


# NOTE: dependent on your Keras version
#       this script used 2.1.1
# [<keras.engine.topology.InputLayer at 0x112fe4358>,
#  <keras.layers.convolutional.Conv2D at 0x112fe46a0>,
#  <keras.layers.normalization.BatchNormalization at 0x112fe4630>,
#  <keras.layers.core.Activation at 0x112fe4eb8>,
#  <keras.layers.pooling.MaxPooling2D at 0x10ed4be48>,
#  <keras.layers.convolutional.Conv2D at 0x1130723c8>,
#  <keras.layers.normalization.BatchNormalization at 0x113064710>,
#  <keras.layers.core.Activation at 0x113092dd8>,
#  <keras.layers.convolutional.Conv2D at 0x11309e908>,
#  <keras.layers.normalization.BatchNormalization at 0x11308a550>,
#  <keras.layers.core.Activation at 0x11312ac88>,
#  <keras.layers.convolutional.Conv2D at 0x1131207b8>,
#  <keras.layers.convolutional.Conv2D at 0x1131b8da0>,
#  <keras.layers.normalization.BatchNormalization at 0x113115550>,
#  <keras.layers.normalization.BatchNormalization at 0x1131a01d0>,
#  <keras.layers.merge.Add at 0x11322f0f0>,
#  <keras.layers.core.Activation at 0x113246cf8>]


# define some additional layers so they have a forward function
class ReLULayer:
  def forward(self, X):
    return tf.nn.relu(X)

  def get_params(self):
    return []

class MaxPoolLayer:
  def __init__(self, dim):
    self.dim = dim

  def forward(self, X):
    return tf.nn.max_pool(
      X,
      ksize=[1, self.dim, self.dim, 1],
      strides=[1, 2, 2, 1],
      padding='VALID'
    )

  def get_params(self):
    return []

class PartialResNet:
  def __init__(self):
    self.layers = [
      # before conv block
      ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
      BatchNormLayer(64),
      ReLULayer(),
      MaxPoolLayer(dim=3),
      # conv block
      ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
    ]
    self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    self.output = self.forward(self.input_)

  def copyFromKerasLayers(self, layers):
    self.layers[0].copyFromKerasLayers(layers[1])
    self.layers[1].copyFromKerasLayers(layers[2])
    self.layers[4].copyFromKerasLayers(layers[5:])

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def predict(self, X):
    assert(self.session is not None)
    return self.session.run(
      self.output,
      feed_dict={self.input_: X}
    )

  def set_session(self, session):
    self.session = session
    self.layers[0].session = session
    self.layers[1].session = session
    self.layers[4].set_session(session)

  def get_params(self):
    params = []
    for layer in self.layers:
      params += layer.get_params()


if __name__ == '__main__':
  # you can also set weights to None, it doesn't matter
  resnet = ResNet50(weights='imagenet')

  # you can determine the correct layer
  # by looking at resnet.layers in the console
  partial_model = Model(
    inputs=resnet.input,
    outputs=resnet.layers[16].output
  )
  print(partial_model.summary())
  # for layer in partial_model.layers:
  #   layer.trainable = False

  my_partial_resnet = PartialResNet()

  # make a fake image
  X = np.random.random((1, 224, 224, 3))

  # get keras output
  keras_output = partial_model.predict(X)

  # get my model output
  init = tf.variables_initializer(my_partial_resnet.get_params())

  # note: starting a new session messes up the Keras model
  session = keras.backend.get_session()
  my_partial_resnet.set_session(session)
  session.run(init)

  # first, just make sure we can get any output
  first_output = my_partial_resnet.predict(X)
  print("first_output.shape:", first_output.shape)

  # copy params from Keras model
  my_partial_resnet.copyFromKerasLayers(partial_model.layers)

  # compare the 2 models
  output = my_partial_resnet.predict(X)
  diff = np.abs(output - keras_output).sum()
  if diff < 1e-10:
    print("Everything's great!")
  else:
    print("diff = %s" % diff)
