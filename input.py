"""Builds the Diabetic_Retinopathy_Detection network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import tensorflow as tf
import math
sys.dont_write_bytecode = True

def read_drd(filename_queue):
    """
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result
        width: number of columns in the result
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class DRDrecord(object):
        pass

    result = DRDrecord()

    # input format.
    result.height = 256
    result.width = 256
    result.depth = 3

    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)
    value = tf.parse_single_example(
        value,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        })
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value['image'], tf.float32)
    # record_bytes.set_shape([32*32*3])
    record_bytes = tf.reshape(record_bytes, [result.height, result.width, 3])
    # Store our label to result.label and convert to int32
    result.label = tf.cast(value['label'], tf.int32)
    result.uint8image = record_bytes

    return result


def _generate_image_and_label_batch(image, label,
                                    params, shuffle):
  """
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=params["BATCH_SIZE"],
        num_threads=params["BATCH_SIZE"],
        capacity=params["CAPACITY"]+3*params["BATCH_SIZE"],
        min_after_dequeue=params["CAPACITY"])
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=params["BATCH_SIZE"],
        num_threads=num_preprocess_threads,
        capacity=params["CAPACITY"]+3*params["BATCH_SIZE"])

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  tf.summary.scalar('label', label)

  return images, tf.reshape(label_batch, [params["BATCH_SIZE"]])

def get_filenames(is_training, params):
  if is_training:
    return [os.path.join(params["DATA_DIR"], params["TRAIN_FILE_NAME"])]
  else:
    return [os.path.join(params["DATA_DIR"], params["VALIDATION_FILE_NAME"])]

def get_test_filenames(params):
    return [os.path.join(params["DATA_DIR"], '_test.tfrecords')]

def distorted_inputs(params):
    """
    Args:
      data_dir: Path to the drd data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    is_training = True
    filenames = get_filenames(is_training, params)

    # #sppecifying angles for images to be rotated by
    # number_of_samples =

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    # Read examples from files in the filename queue
    print("the filename queue is {}".format(filename_queue))
    read_input = read_drd(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    height = params["IMAGE_SIZE"]
    width = params["IMAGE_SIZE"]
    NUM_CHANNELS = 3
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, height + 8, width + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [height, width, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    angles = tf.random_uniform([1], -15, 15, dtype=tf.float32, seed=0)
    image = tf.contrib.image.rotate(image, angles * math.pi / 360, interpolation='NEAREST', name=None)

    #scale image
    image = image / 255.0
    image = image -0.5

    print('Filling queue with %d DRD images before starting to train. '
          'This will take a few minutes.' % params["CAPACITY"])

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label, params,shuffle=True)


def inputs(params):
    """
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path
       to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    is_training = False
    filenames = get_filenames(is_training, params)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    print("the filename queue is {}".format(filename_queue))
    read_input = read_drd(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    #scale image
    image = image/255.0
    image = image - 0.5

    print('Filling queue with %d DRD images before starting to train. '
          'This will take a few minutes.' % params["CAPACITY"])

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label, params,shuffle=False)

def test_inputs(params, batch_size=1):
    """Construct input for SVHN evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path
       to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = get_test_filenames(params["DATA_DIR"])

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    print("the filename queue is {}".format(filename_queue))
    read_input = read_drd(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    print('Filling queue with %d DRD images before starting to train. '
          'This will take a few minutes.' % params["CAPACITY"])

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label, params,shuffle=False)
