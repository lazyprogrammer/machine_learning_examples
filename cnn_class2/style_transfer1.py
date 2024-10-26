# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

# In this script, we will focus on generating the content
# E.g. given an image, can we recreate the same image

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b


import tensorflow as tf
#if tf.__version__.startswith('2'):
#  tf.compat.v1.disable_eager_execution()


def VGG16_AvgPool(shape):
  # we want to account for features across the entire image
  # so get rid of the maxpool which throws away information
  vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
  vgg_clone = clone_model(vgg)

  # new_model = Sequential()
  # for layer in vgg.layers:
  #   if layer.__class__ == MaxPooling2D:
  #     # replace it with average pooling
  #     new_model.add(AveragePooling2D())
  #   else:
  #     new_model.add(layer)

  #i = vgg.input
  #x = i
  for layer in vgg_clone.layers:
    if isinstance(layer, MaxPooling2D):
      # replace it with average pooling
      layer = AveragePooling2D(pool_size=layer.pool_size)
    #else:
    #  x = layer(x)

# return Model(i, x)
  return vgg_clone

def VGG16_AvgPool_CutOff(shape, num_convs):
  # there are 13 convolutions in total
  # we can pick any of them as the "output"
  # of our content model

  if num_convs < 1 or num_convs > 13:
    print("num_convs must be in the range [1, 13]")
    return None

  model = VGG16_AvgPool(shape)
  # new_model = Sequential()
  # n = 0
  # for layer in model.layers:
  #   if layer.__class__ == Conv2D:
  #     n += 1
  #   new_model.add(layer)
  #   if n >= num_convs:
  #     break

  n = 0
  output = None
  for layer in model.layers:
    if layer.__class__ == Conv2D:
      n += 1
    if n >= num_convs:
      output = layer.output
      break

  return Model(model.input, output)


def unpreprocess(img):
  img[..., 0] += 103.939
  img[..., 1] += 116.779
  img[..., 2] += 126.68
  img = img[..., ::-1]
  return img


def scale_img(x):
  x = x - x.min()
  x = x / x.max()
  return x


if __name__ == '__main__':

  # open an image
  # feel free to try your own
  # path = '../large_files/caltech101/101_ObjectCategories/elephant/image_0002.jpg'
  path = '.\\cnn_class2\\content\\elephant.jpg'
  img = image.load_img(path)

  # convert image to array and preprocess for vgg
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  # we'll use this throughout the rest of the script
  batch_shape = x.shape
  shape = x.shape[1:]

  # see the image
  plt.imshow(img)
  plt.show()


  # make a content model
  # try different cutoffs to see the images that result
  content_model = VGG16_AvgPool_CutOff(shape, 11)

  # make the target
  target = K.variable(content_model.predict(x))


  # try to match the image

  # define our loss in keras
  #loss_layer = Lambda(lambda inputs: K.mean(K.square(inputs[0] - inputs[1])))
  #loss = loss_layer([target, content_model.output])

  def get_loss_and_grads(inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        # Compute the loss as the mean squared difference between target and model output
        loss_value = K.mean(K.square(target - content_model(inputs)))
    # Compute the gradient of loss with respect to the inputs
    grads_value = tape.gradient(loss_value, inputs)
    return loss_value, grads_value



  def get_loss_and_grads_wrapper(x_vec):
    # scipy's minimizer allows us to pass back
    # function value f(x) and its gradient f'(x)
    # simultaneously, rather than using the fprime arg
    #
    # we cannot use get_loss_and_grads() directly
    # input to minimizer func must be a 1-D array
    # input to get_loss_and_grads must be [batch_of_images]
    #
    # gradient must also be a 1-D array
    # and both loss and gradient must be np.float64
    # will get an error otherwise
    x_tensor = tf.convert_to_tensor(x_vec.reshape(*batch_shape), dtype=tf.float32)
    l, g = get_loss_and_grads(x_tensor)
    #l, g = get_loss_and_grads(x_vec.reshape(*batch_shape))
    return l.numpy().astype(np.float64), g.numpy().flatten().astype(np.float64)



  from datetime import datetime
  t0 = datetime.now()
  losses = []
  x = np.random.randn(np.prod(batch_shape))
  for i in range(10):
    x, l, _ = fmin_l_bfgs_b(
      func=get_loss_and_grads_wrapper,
      x0=x,
      # bounds=[[-127, 127]]*len(x.flatten()),
      maxfun=20
    )
    x = np.clip(x, -127, 127)
    # print("min:", x.min(), "max:", x.max())
    print("iter=%s, loss=%s" % (i, l))
    losses.append(l)

  print("duration:", datetime.now() - t0)
  plt.plot(losses)
  plt.show()

  newimg = x.reshape(*batch_shape)
  final_img = unpreprocess(newimg)


  plt.imshow(scale_img(final_img[0]))
  plt.show()
