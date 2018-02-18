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



class PartialResNet:
  def __init__(self):
    # TODO
    pass

  def copyFromKerasLayers(self, layers):
    # TODO
    pass

  def predict(self, X):
    # TODO
    pass

  def set_session(self, session):
    self.session = session
    # TODO: finish this

  def get_params(self):
    params = []
    # TODO: finish this


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
