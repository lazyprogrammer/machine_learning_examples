# https://deeplearningcourses.com/c/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image

import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from collections import Counter


# get the data from: http://vision.ucsd.edu/content/yale-face-database
files = glob('../large_files/yalefaces/subject*')

# easier to randomize later
np.random.shuffle(files)

# number of samples
N = len(files)


def load_img(filepath):
  # load image and downsample
  img = image.img_to_array(image.load_img(filepath, target_size=[60, 80])).astype('uint8')
  return img



# look at an image for fun
img = load_img(np.random.choice(files))
plt.imshow(img)
plt.show()


# try load images as arrays
# yes, I cheated and checked beforehand that all the images were the same shape!
shape = [N] + list(img.shape)
images = np.zeros(shape)
for i, f in enumerate(files):
  # img = image.img_to_array(image.load_img(f)).astype('uint8')
  img = load_img(f)
  images[i] = img


# make the labels
# all the filenames are something like 'subject13.happy'
labels = np.zeros(N)
for i, f in enumerate(files):
  filename = f.rsplit('/', 1)[-1]
  subject_num = filename.split('.', 1)[0]

  # subtract 1 since the filenames start from 1
  idx = int(subject_num.replace('subject', '')) - 1
  labels[i] = idx


# how many of each subject do we have?
label_count = Counter(labels)

# set of unique labels
unique_labels = set(label_count.keys())

# get the number of subjects
n_subjects = len(label_count)

# let's make it so 3 images for each subject are test data
# number of test points is then
n_test = 3 * n_subjects
n_train = N - n_test


# initialize arrays to hold train and test images
train_images = np.zeros([n_train] + list(img.shape))
train_labels = np.zeros(n_train)
test_images = np.zeros([n_test] + list(img.shape))
test_labels = np.zeros(n_test)


count_so_far = {}
train_idx = 0
test_idx = 0
for img, label in zip(images, labels):
  # increment the count
  count_so_far[label] = count_so_far.get(label, 0) + 1

  if count_so_far[label] > 3:
    # we have already added 3 test images for this subject
    # so add the rest to train
    train_images[train_idx] = img
    train_labels[train_idx] = label
    train_idx += 1

  else:
    # add the first 3 images to test
    test_images[test_idx] = img
    test_labels[test_idx] = label
    test_idx += 1


# create label2idx mapping for easy access
train_label2idx = {}
test_label2idx = {}

for i, label in enumerate(train_labels):
  if label not in train_label2idx:
    train_label2idx[label] = [i]
  else:
    train_label2idx[label].append(i)

for i, label in enumerate(test_labels):
  if label not in test_label2idx:
    test_label2idx[label] = [i]
  else:
    test_label2idx[label].append(i)


# come up with all possible training sample indices
train_positives = []
train_negatives = []
test_positives = []
test_negatives = []

for label, indices in train_label2idx.items():
  # all indices that do NOT belong to this subject
  other_indices = set(range(n_train)) - set(indices)

  for i, idx1 in enumerate(indices):
    for idx2 in indices[i+1:]:
      train_positives.append((idx1, idx2))

    for idx2 in other_indices:
      train_negatives.append((idx1, idx2))

for label, indices in test_label2idx.items():
  # all indices that do NOT belong to this subject
  other_indices = set(range(n_test)) - set(indices)

  for i, idx1 in enumerate(indices):
    for idx2 in indices[i+1:]:
      test_positives.append((idx1, idx2))

    for idx2 in other_indices:
      test_negatives.append((idx1, idx2))


batch_size = 64
def train_generator():
  # for each batch, we will send 1 pair of each subject
  # and the same number of non-matching pairs
  n_batches = int(np.ceil(len(train_positives) / batch_size))
  
  while True:
    np.random.shuffle(train_positives)

    n_samples = batch_size * 2
    shape = [n_samples] + list(img.shape)
    x_batch_1 = np.zeros(shape)
    x_batch_2 = np.zeros(shape)
    y_batch = np.zeros(n_samples)

    for i in range(n_batches):
      pos_batch_indices = train_positives[i * batch_size: (i + 1) * batch_size]

      # fill up x_batch and y_batch
      j = 0
      for idx1, idx2 in pos_batch_indices:
        x_batch_1[j] = train_images[idx1]
        x_batch_2[j] = train_images[idx2]
        y_batch[j] = 1 # match
        j += 1

      # get negative samples
      neg_indices = np.random.choice(len(train_negatives), size=len(pos_batch_indices), replace=False)
      for neg in neg_indices:
        idx1, idx2 = train_negatives[neg]
        x_batch_1[j] = train_images[idx1]
        x_batch_2[j] = train_images[idx2]
        y_batch[j] = 0 # non-match
        j += 1

      x1 = x_batch_1[:j]
      x2 = x_batch_2[:j]
      y = y_batch[:j]
      yield [x1, x2], y


# same thing as the train generator except no shuffling and it uses the test set
def test_generator():
  n_batches = int(np.ceil(len(test_positives) / batch_size))

  while True:
    n_samples = batch_size * 2
    shape = [n_samples] + list(img.shape)
    x_batch_1 = np.zeros(shape)
    x_batch_2 = np.zeros(shape)
    y_batch = np.zeros(n_samples)

    for i in range(n_batches):
      pos_batch_indices = test_positives[i * batch_size: (i + 1) * batch_size]

      # fill up x_batch and y_batch
      j = 0
      for idx1, idx2 in pos_batch_indices:
        x_batch_1[j] = test_images[idx1]
        x_batch_2[j] = test_images[idx2]
        y_batch[j] = 1 # match
        j += 1

      # get negative samples
      neg_indices = np.random.choice(len(test_negatives), size=len(pos_batch_indices), replace=False)
      for neg in neg_indices:
        idx1, idx2 = test_negatives[neg]
        x_batch_1[j] = test_images[idx1]
        x_batch_2[j] = test_images[idx2]
        y_batch[j] = 0 # non-match
        j += 1

      x1 = x_batch_1[:j]
      x2 = x_batch_2[:j]
      y = y_batch[:j]
      yield [x1, x2], y




# build the base neural network
i = Input(shape=img.shape)
x = Conv2D(filters=32, kernel_size=(3, 3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=(3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=50)(x) # feature vector

cnn = Model(inputs=i, outputs=x)


# feed both images into the same CNN
img_placeholder1 = Input(shape=img.shape)
img_placeholder2 = Input(shape=img.shape)

# get image features
feat1 = cnn(img_placeholder1)
feat2 = cnn(img_placeholder2)


# calculate the Euclidean distance between feature 1 and feature 2
def euclidean_distance(features):
  x, y = features
  return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# lambda layer to output distance between feat1 and feat2
dist_layer = Lambda(euclidean_distance)([feat1, feat2])


# the model we will actually train
model = Model(inputs=[img_placeholder1, img_placeholder2], outputs=dist_layer)


# loss function for siamese network
def contrastive_loss(y_true, y_pred):
  margin = 1
  return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# compile the model
model.compile(
  loss=contrastive_loss,
  optimizer='adam',
)


# calculate accuracy before training
# since the dataset is imbalanced, we'll report tp, tn, fp, fn
def get_train_accuracy(threshold=0.85):
  positive_distances = []
  negative_distances = []

  tp = 0
  tn = 0
  fp = 0
  fn = 0

  batch_size = 64
  x_batch_1 = np.zeros([batch_size] + list(img.shape))
  x_batch_2 = np.zeros([batch_size] + list(img.shape))
  n_batches = int(np.ceil(len(train_positives) / batch_size))
  for i in range(n_batches):
    print(f"pos batch: {i+1}/{n_batches}")
    pos_batch_indices = train_positives[i * batch_size: (i + 1) * batch_size]

    # fill up x_batch and y_batch
    j = 0
    for idx1, idx2 in pos_batch_indices:
      x_batch_1[j] = train_images[idx1]
      x_batch_2[j] = train_images[idx2]
      j += 1

    x1 = x_batch_1[:j]
    x2 = x_batch_2[:j]
    distances = model.predict([x1, x2]).flatten()
    positive_distances += distances.tolist()

    # update tp, tn, fp, fn
    tp += (distances < threshold).sum()
    fn += (distances > threshold).sum()

  n_batches = int(np.ceil(len(train_negatives) / batch_size))
  for i in range(n_batches):
    print(f"neg batch: {i+1}/{n_batches}")
    neg_batch_indices = train_negatives[i * batch_size: (i + 1) * batch_size]

    # fill up x_batch and y_batch
    j = 0
    for idx1, idx2 in neg_batch_indices:
      x_batch_1[j] = train_images[idx1]
      x_batch_2[j] = train_images[idx2]
      j += 1

    x1 = x_batch_1[:j]
    x2 = x_batch_2[:j]
    distances = model.predict([x1, x2]).flatten()
    negative_distances += distances.tolist()

    # update tp, tn, fp, fn
    fp += (distances < threshold).sum()
    tn += (distances > threshold).sum()

  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  print(f"sensitivity (tpr): {tpr}, specificity (tnr): {tnr}")

  plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
  plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
  plt.legend()
  plt.show()



def get_test_accuracy(threshold=0.85):
  positive_distances = []
  negative_distances = []

  tp = 0
  tn = 0
  fp = 0
  fn = 0

  batch_size = 64
  x_batch_1 = np.zeros([batch_size] + list(img.shape))
  x_batch_2 = np.zeros([batch_size] + list(img.shape))
  n_batches = int(np.ceil(len(test_positives) / batch_size))
  for i in range(n_batches):
    print(f"pos batch: {i+1}/{n_batches}")
    pos_batch_indices = test_positives[i * batch_size: (i + 1) * batch_size]

    # fill up x_batch and y_batch
    j = 0
    for idx1, idx2 in pos_batch_indices:
      x_batch_1[j] = test_images[idx1]
      x_batch_2[j] = test_images[idx2]
      j += 1

    x1 = x_batch_1[:j]
    x2 = x_batch_2[:j]
    distances = model.predict([x1, x2]).flatten()
    positive_distances += distances.tolist()

    # update tp, tn, fp, fn
    tp += (distances < threshold).sum()
    fn += (distances > threshold).sum()

  n_batches = int(np.ceil(len(test_negatives) / batch_size))
  for i in range(n_batches):
    print(f"neg batch: {i+1}/{n_batches}")
    neg_batch_indices = test_negatives[i * batch_size: (i + 1) * batch_size]

    # fill up x_batch and y_batch
    j = 0
    for idx1, idx2 in neg_batch_indices:
      x_batch_1[j] = test_images[idx1]
      x_batch_2[j] = test_images[idx2]
      j += 1

    x1 = x_batch_1[:j]
    x2 = x_batch_2[:j]
    distances = model.predict([x1, x2]).flatten()
    negative_distances += distances.tolist()

    # update tp, tn, fp, fn
    fp += (distances < threshold).sum()
    tn += (distances > threshold).sum()


  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  print(f"sensitivity (tpr): {tpr}, specificity (tnr): {tnr}")

  plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
  plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
  plt.legend()
  plt.show()




# params for training
train_steps = int(np.ceil(len(train_positives) * 2 / batch_size))
valid_steps = int(np.ceil(len(test_positives) * 2 / batch_size))

# fit the model
r = model.fit_generator(
  train_generator(),
  steps_per_epoch=train_steps,
  epochs=20,
  validation_data=test_generator(),
  validation_steps=valid_steps,
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

get_train_accuracy()
get_test_accuracy()
