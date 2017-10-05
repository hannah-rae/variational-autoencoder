import tensorflow as tf
import sys
from time import strftime, gmtime

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('input_rows', 144, 'Number of rows in input images')
flags.DEFINE_integer('input_cols', 160, 'Number of columns in input images')
flags.DEFINE_integer('input_filters', 3, 'Number of filters in input images')
flags.DEFINE_integer('num_train_samples', 42096, 'Number of training examples')
flags.DEFINE_integer('n_z', 2200, 'Size of bottleneck layer')
flags.DEFINE_integer('C1', 64, 'Number of feature maps in first conv layer')
flags.DEFINE_integer('C2', 32, 'Number of feature maps in second conv layer')
flags.DEFINE_integer('C3', 16, 'Number of feature maps in third conv layer')
flags.DEFINE_integer('FC4', 1024, 'Number of feature maps in first fully connected layer')
flags.DEFINE_integer('FC5', 512, 'Number of feature maps in second fully connected layer')
flags.DEFINE_integer('FC6', 256, 'Number of feature maps in third fully connected layer')
flags.DEFINE_integer('kernel_size', 3, 'Kernel size for all convolutions')
flags.DEFINE_integer('stride_size', 1, 'Stride size for all convolutions')
flags.DEFINE_string('log_dir', '/Users/hannahrae/src/autoencoder/MastcamVAE/log', 'Directory to store training checkpoints')
flags.DEFINE_string('train_dir', '/Users/hannahrae/data/mcam_vae/train', 'Directory containing training images')
flags.DEFINE_string('test_dir', '/Users/hannahrae/data/mcam_vae/test', 'Directory containing test images')
flags.DEFINE_string('val_dir', '/Users/hannahrae/data/mcam_vae/validation', 'Directory containing validation images')
flags.DEFINE_integer('max_steps', 1000, 'Maximum number of times to run trainer')
flags.DEFINE_integer('num_epochs', 10, 'Number of times to go through all the training data')
flags.DEFINE_float('l2_beta', 0.01, 'Beta term in L2 regularization of weights')
flags.DEFINE_string('tmp_dir', './tmp/data', 'Directory to download data files and write the converted result')
flags.DEFINE_string('train_records', '/Users/hannahrae/src/autoencoder/MastcamVAE/tmp/data/train.tfrecords', 'Directory on local machine or GCS where TFRecords for training located')
flags.DEFINE_string('test_records', 'gs://mcamvae/test.tfrecords', 'Directory on local machine or GCS where TFRecords for testing located')
flags.DEFINE_string('eval_records', 'gs://mcamvae/validation.tfrecords', 'Directory on local machine or GCS where TFRecords for validation located')
flags.DEFINE_integer('eval_steps', 100, 'Number of training steps before evaluating validation set')
flags.DEFINE_string('job-dir', 'gs://MastcamVAE/output', 'GCS job directory')

FLAGS = flags.FLAGS