# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

# In this script, we will focus on generating an image
# that attempts to match the content of one input image
# and the style of another input image.
#
# We accomplish this by balancing the content loss
# and style loss simultaneously.

from tensorflow.keras.layers import Layer #type: ignore #Input, Lambda, Dense, Flatten
# from keras.layers import AveragePooling2D, MaxPooling2D
# from keras.layers.convolutional import Conv2D
from tensorflow.keras.models import Model #type: ignore
# from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
#from skimage.transform import resize

import tensorflow.keras.backend as K #type: ignore
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from style_transfer1 import VGG16_AvgPool, scale_img
from style_transfer2 import style_loss, minimize
#from scipy.optimize import fmin_l_bfgs_b


# load the content image
def load_img_and_preprocess(path, shape=None):
  img = image.load_img(path, target_size=shape)

  # convert image to array and preprocess for vgg
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  return x



content_img = load_img_and_preprocess(
  # '../large_files/caltech101/101_ObjectCategories/elephant/image_0002.jpg',
  # 'batman.jpg',
  '.\\cnn_class2\\content\\sydney.jpg',
  # (225, 300),
)

# resize the style image
# since we don't care too much about warping it
h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess(
  # 'styles/starrynight.jpg',
  # 'styles/flowercarrier.jpg',
  # 'styles/monalisa.jpg',
  '.\\cnn_class2\\styles\\lesdemoisellesdavignon.jpg',
  (h, w)
)


# we'll use this throughout the rest of the script
batch_shape = content_img.shape
shape = content_img.shape[1:]


# we want to make only 1 VGG here
# as you'll see later, the final model needs
# to have a common input
vgg = VGG16_AvgPool(shape)


# create the content model
# we only want 1 output
# remember you can call vgg.summary() to see a list of layers
# 1,2,4,5,7-9,11-13,15-17
content_model = Model(vgg.input, vgg.layers[13].output)
content_target = tf.Variable(content_model.predict(content_img))


# create the style model
# we want multiple outputs
# we will take the same approach as in style_transfer2.py
symbolic_conv_outputs = [
  vgg.get_layer(layer.name).output for layer in vgg.layers
    if layer.name.endswith('conv1')
]

# make a big model that outputs multiple layers' outputs
style_model = Model(vgg.input, symbolic_conv_outputs)

# calculate the targets that are output at each layer
style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

# we will assume the weight of the content loss is 1
# and only weight the style losses
style_weights = [0.2,0.4,0.3,0.5,0.2]



# create the total loss which is the sum of content + style loss
#loss = K.mean(K.square(content_model.output - content_target))

class ContentLossLayer(Layer):
    def __init__(self, content_target, **kwargs):
        super(ContentLossLayer, self).__init__(**kwargs)
        self.content_target = content_target

    def call(self, inputs):
        return tf.reduce_mean(tf.square(inputs - self.content_target))

with tf.GradientTape() as tape:
  # Instantiate the content loss layer
  content_loss_layer = ContentLossLayer(content_target)

  # Now compute the loss
  loss = content_loss_layer(content_model.output)

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
  # gram_matrix() expects a (H, W, C) as input
  loss += w * style_loss(symbolic[0], actual[0])


# once again, create the gradients and loss + grads function
# note: it doesn't matter which model's input you use
# they are both pointing to the same keras Input layer in memory
grads = tape.gradient(loss, vgg.input)

# just like theano.function
get_loss_and_grads = K.function(
  inputs=[vgg.input],
  outputs=[loss] + grads
)


def get_loss_and_grads_wrapper(x_vec):
  l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
  return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()
