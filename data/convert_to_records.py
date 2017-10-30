"""Converts Mastcam thumbnail data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append("/Users/hannahrae/src/autoencoder/MastcamVAE/") # for local machine

import argparse
import os

from utils.flags import FLAGS
import tensorflow as tf
import data_sets


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, name):
  """Converts a dataset to tfrecords."""
  num_examples = images.shape[0]
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.tmp_dir, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # Get the data.
  datasets = data_sets.DataSet(input_rows=FLAGS.input_rows, input_cols=FLAGS.input_cols, num_filters=FLAGS.input_filters)
  # Convert to Examples and write the result to TFRecords.
  convert_to(datasets.train, 'train')
  #convert_to(datasets.validation, 'validation')
  convert_to(datasets.test, 'test')


if __name__ == '__main__':
  tf.app.run()