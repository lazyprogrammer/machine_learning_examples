# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_resnet_convblock import ConvLayer, BatchNormLayer


class IdentityBlock:
  def __init__(self, mi, fm_sizes, activation=tf.nn.relu):
    # conv1, conv2, conv3
    # note: # feature maps shortcut = # feauture maps conv 3
    assert(len(fm_sizes) == 3)

    # note: kernel size in 2nd conv is always 3
    #       so we won't bother including it as an arg

    self.session = None
    self.f = tf.nn.relu
    
    # init main branch
    # Conv -> BN -> F() ---> Conv -> BN -> F() ---> Conv -> BN
    self.conv1 = ConvLayer(1, mi, fm_sizes[0], 1)
    self.bn1   = BatchNormLayer(fm_sizes[0])
    self.conv2 = ConvLayer(3, fm_sizes[0], fm_sizes[1], 1, 'SAME')
    self.bn2   = BatchNormLayer(fm_sizes[1])
    self.conv3 = ConvLayer(1, fm_sizes[1], fm_sizes[2], 1)
    self.bn3   = BatchNormLayer(fm_sizes[2])

    # in case needed later
    self.layers = [
      self.conv1, self.bn1,
      self.conv2, self.bn2,
      self.conv3, self.bn3,
    ]

    # this will not be used when input passed in from
    # a previous layer
    self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, mi))
    self.output = self.forward(self.input_)

  def forward(self, X):
    # main branch
    FX = self.conv1.forward(X)
    FX = self.bn1.forward(FX)
    FX = self.f(FX)
    FX = self.conv2.forward(FX)
    FX = self.bn2.forward(FX)
    FX = self.f(FX)
    FX = self.conv3.forward(FX)
    FX = self.bn3.forward(FX)

    return self.f(FX + X)

  def predict(self, X):
    assert(self.session is not None)
    return self.session.run(
      self.output,
      feed_dict={self.input_: X}
    )

  def set_session(self, session):
    # need to make this a session
    # so assignment happens on sublayers too
    self.session = session
    self.conv1.session = session
    self.bn1.session = session
    self.conv2.session = session
    self.bn2.session = session
    self.conv3.session = session
    self.bn3.session = session

  def copyFromKerasLayers(self, layers):
    assert(len(layers) == 10)
    # <keras.layers.convolutional.Conv2D at 0x7fa44255ff28>,
    # <keras.layers.normalization.BatchNormalization at 0x7fa44250e7b8>,
    # <keras.layers.core.Activation at 0x7fa44252d9e8>,
    # <keras.layers.convolutional.Conv2D at 0x7fa44253af60>,
    # <keras.layers.normalization.BatchNormalization at 0x7fa4424e4f60>,
    # <keras.layers.core.Activation at 0x7fa442494828>,
    # <keras.layers.convolutional.Conv2D at 0x7fa4424a2da0>,
    # <keras.layers.normalization.BatchNormalization at 0x7fa44244eda0>,
    # <keras.layers.merge.Add at 0x7fa44245d5c0>,
    # <keras.layers.core.Activation at 0x7fa44240aba8>
    self.conv1.copyFromKerasLayers(layers[0])
    self.bn1.copyFromKerasLayers(layers[1])
    self.conv2.copyFromKerasLayers(layers[3])
    self.bn2.copyFromKerasLayers(layers[4])
    self.conv3.copyFromKerasLayers(layers[6])
    self.bn3.copyFromKerasLayers(layers[7])

  def get_params(self):
    params = []
    for layer in self.layers:
      params += layer.get_params()
    return params


if __name__ == '__main__':
  identity_block = IdentityBlock(mi=256, fm_sizes=[64, 64, 256])

  # make a fake image
  X = np.random.random((1, 224, 224, 256))

  init = tf.global_variables_initializer()
  with tf.Session() as session:
    identity_block.set_session(session)
    session.run(init)

    output = identity_block.predict(X)
    print("output.shape:", output.shape)
