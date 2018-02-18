# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

# simple script to adapt object detection notebook from
# https://github.com/tensorflow/models
# to work on videos
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os, sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import imageio

if tf.__version__ < '1.4.0':
  raise ImportError(
    'Please upgrade your tensorflow installation to v1.4.* or later!'
  )


# change this to wherever you cloned the tensorflow models repo
# which I assume you've already downloaded from:
# https://github.com/tensorflow/models
RESEARCH_PATH = '../../tf-models/research'
MODELS_PATH = '../../tf-models/research/object_detection'
sys.path.append(RESEARCH_PATH)
sys.path.append(MODELS_PATH)

# import local modules
import object_detection
from utils import label_map_util
from utils import visualization_utils as vis_util


# I've assumed you already ran the notebook and downloaded the model
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = '%s/%s/frozen_inference_graph.pb' % (MODELS_PATH, MODEL_NAME)
PATH_TO_LABELS = '%s/data/mscoco_label_map.pbtxt' % MODELS_PATH
NUM_CLASSES = 90


# load the model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("categories:")
print(categories)


# convert image -> numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# do some object detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # instead of looping through test images, we'll now loop
    # through our video!

    # get the videos from:
    # https://lazyprogrammer.me/cnn_class2_videos.zip
    # and put them into the same folder as this file

    # open the video
    # input_video = 'catdog'
    # input_video = 'safari'
    input_video = 'traffic'
    video_reader = imageio.get_reader('%s.mp4' % input_video)
    video_writer = imageio.get_writer('%s_annotated.mp4' % input_video, fps=10)

    # loop through and process each frame
    t0 = datetime.now()
    n_frames = 0
    for frame in video_reader:
      # rename for convenience
      image_np = frame
      n_frames += 1

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      # instead of plotting image, we write the frame to video
      video_writer.append_data(image_np)

    fps = n_frames / (datetime.now() - t0).total_seconds()
    print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

    # clean up
    video_writer.close()
