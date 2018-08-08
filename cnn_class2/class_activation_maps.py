# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from glob import glob



# get the image files
# http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# http://www.vision.caltech.edu/Image_Datasets/Caltech256/
image_files = glob('../large_files/256_ObjectCategories/*/*.jp*g')
image_files += glob('../large_files/101_ObjectCategories/*/*.jp*g')



# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()


# add preprocessing layer to the front of VGG
resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

# view the structure of the model
# if you want to confirm we need activation_49
resnet.summary()

# make a model to get output before flatten
activation_layer = resnet.get_layer('activation_49')

# create a model object
model = Model(inputs=resnet.input, outputs=activation_layer.output)

# get the feature map weights
final_dense = resnet.get_layer('fc1000')
W = final_dense.get_weights()[0]


while True:
  img = image.load_img(np.random.choice(image_files), target_size=(224, 224))
  x = preprocess_input(np.expand_dims(img, 0))
  fmaps = model.predict(x)[0] # 7 x 7 x 2048

  # get predicted class
  probs = resnet.predict(x)
  classnames = decode_predictions(probs)[0]
  print(classnames)
  classname = classnames[0][1]
  pred = np.argmax(probs[0])

  # get the 2048 weights for the relevant class
  w = W[:, pred]

  # "dot" w with fmaps
  cam = fmaps.dot(w)

  # upsample to 224 x 224
  # 7 x 32 = 224
  cam = sp.ndimage.zoom(cam, (32, 32), order=1)

  plt.subplot(1,2,1)
  plt.imshow(img, alpha=0.8)
  plt.imshow(cam, cmap='jet', alpha=0.5)
  plt.subplot(1,2,2)
  plt.imshow(img)
  plt.title(classname)
  plt.show()

  ans = input("Continue? (Y/n)")
  if ans and ans[0].lower() == 'n':
    break



# def slowversion(A, w):
#   N = len(w)
#   result = np.zeros(A.shape[:-1])
#   for i in range(N):
#     result += A[:,:,i]*w[i]
#   return result

