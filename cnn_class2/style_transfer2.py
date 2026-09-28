# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

# In this script, we will focus on generating an image
# with the same style as the input image.
# But NOT the same content.
# It should capture only the essence of the style.

from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
#from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda #type:ignore

from style_transfer1 import VGG16_AvgPool, unpreprocess, scale_img
# from skimage.transform import resize
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K #type: ignore



# def gram_matrix(img):
#   # input is (H, W, C) (C = # feature maps)
#   # we first need to convert it to (C, H*W)
#   X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
  
#   # now, calculate the gram matrix
#   # gram = XX^T / N
#   # the constant is not important since we'll be weighting these
#   G = K.dot(X, K.transpose(X))/img.get_shape().num_elements()
#   return G

class GramMatrixLayer(Layer):
    def call(self, inputs):
        # Input shape is expected to be (H, W, C)
        # Permute dimensions to (C, H, W)
        permuted_img = tf.transpose(inputs, perm=[2, 0, 1])  # (C, H, W)

        # Flatten the permuted image to (C, H*W)
        flattened_img = tf.reshape(permuted_img, (tf.shape(permuted_img)[0], -1))  # (C, H*W)

        # Calculate the Gram matrix
        num_elements = tf.cast(tf.reduce_prod(K.int_shape(inputs)[1:]), tf.float32) 
        G = K.dot(flattened_img, K.transpose(flattened_img)) / num_elements
        return G

def gram_matrix(img):
    return GramMatrixLayer()(img)


def style_loss(y, t):
  return Lambda(lambda x: K.mean(K.square(gram_matrix(x[0]) - gram_matrix(x[1]))))([y, t])


# let's generalize this and put it into a function
def minimize(fn, epochs, batch_shape):
  t0 = datetime.now()
  losses = []
  x = np.random.randn(np.prod(batch_shape))
  for i in range(epochs):
    x, l, _ = fmin_l_bfgs_b(
      func=fn,
      x0=x,
      maxfun=20
    )
    x = np.clip(x, -127, 127)
    print("iter=%s, loss=%s" % (i, l))
    losses.append(l)

  print("duration:", datetime.now() - t0)
  plt.plot(losses)
  plt.show()

  newimg = x.reshape(*batch_shape)
  final_img = unpreprocess(newimg)
  return final_img[0]


if __name__ == '__main__':
  # try these, or pick your own!
  path = '.\\cnn_class2\\styles\\starrynight.jpg'
  # path = 'styles/flowercarrier.jpg'
  # path = 'styles/monalisa.jpg'
  # path = 'styles/lesdemoisellesdavignon.jpg'


  # load the data
  img = image.load_img(path)

  # convert image to array and preprocess for vgg
  x = image.img_to_array(img)

  # look at the image
  plt.imshow(x)
  plt.show()

  # make it (1, H, W, C)
  x = np.expand_dims(x, axis=0)

  # preprocess into VGG expected format
  x = preprocess_input(x)

  # we'll use this throughout the rest of the script
  batch_shape = x.shape
  shape = x.shape[1:]

  # let's take the first convolution at each block of convolutions
  # to be our target outputs
  # remember that you can print out the model summary if you want
  vgg = VGG16_AvgPool(shape)

  # Note: need to select output at index 1, since outputs at
  # index 0 correspond to the original vgg with maxpool
  symbolic_conv_outputs = [
    vgg.get_layer(layer.name).output for layer in vgg.layers
    if layer.name.endswith('conv1')
  ]

  # pick the earlier layers for
  # a more "localized" representation
  # this is opposed to the content model
  # where the later layers represent a more "global" structure
  # symbolic_conv_outputs = symbolic_conv_outputs[:2]

  # make a big model that outputs multiple layers' outputs
  multi_output_model = Model(vgg.input, symbolic_conv_outputs)

  # calculate the targets that are output at each layer
  style_layers_outputs = [K.variable(y) for y in multi_output_model.predict(x)]

  # calculate the total style loss
  def get_loss_and_grads(inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)  # Ensure it's a tensor
    with tf.GradientTape() as tape:
        tape.watch(inputs)

        # Calculate the total style loss
        loss_value = 0
        for symbolic, actual in zip(symbolic_conv_outputs, style_layers_outputs):
            current_loss = style_loss(symbolic[0], actual[0])
            print(f'Loss: {current_loss.numpy().astype(np.float64)}')
            loss_value += current_loss
    # Compute gradients
    grads_value = tape.gradient(loss_value, inputs)
    return loss_value, grads_value


  def get_loss_and_grads_wrapper(x_vec):
    # Convert the 1-D array back to the appropriate tensor shape
    x_tensor = tf.convert_to_tensor(x_vec.reshape(*batch_shape), dtype=tf.float32)
    
    # Get the loss and gradients
    l, g = get_loss_and_grads(x_tensor)
    
    # Return the loss and the gradients as required by the optimizer
    return l.numpy().astype(np.float64), g.numpy().flatten().astype(np.float64)

  final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
  plt.imshow(scale_img(final_img))
  plt.show()
