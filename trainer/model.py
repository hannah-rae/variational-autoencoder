import sys
sys.path.append("/Users/hannahrae/src/autoencoder/MastcamVAE/")

import tensorflow as tf
import numpy as np
import os
import time
import cv2
from random import shuffle
from utils.ops import *

from utils.flags import FLAGS
import data.data_sets

TRAIN_FILE = FLAGS.train_records
# VALIDATION_FILE = FLAGS.eval_records

class LatentAttention():
    def __init__(self):
        self.writer = tf.summary.FileWriter(makeTrialOutputPath(FLAGS.log_dir))

        self.n_z = FLAGS.n_z
        self.batchsize = FLAGS.batch_size

        self.images = self.inputs(train=True, num_epochs=FLAGS.num_epochs)
        z_mean, z_stddev = self.recognition(self.images)
        self.images = tf.reshape(self.images, [tf.shape(self.images)[0], FLAGS.input_rows*FLAGS.input_cols*FLAGS.input_filters])

        # Sample from Gaussian distribution
        samples = tf.random_normal([tf.shape(self.images)[0], self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [tf.shape(self.images)[0], FLAGS.input_rows*FLAGS.input_cols*FLAGS.input_filters])


        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-6 + generated_flat) + (1-self.images) * tf.log(1e-6 + 1 - generated_flat),1)
        #tf.summary.scalar("generation_loss", self.generation_loss)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        #tf.summary.scalar("latent_loss", self.latent_loss)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        tf.summary.scalar('training/hptuning/metric', self.cost)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.cost)


    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string)
          })

        # Convert from a scalar string tensor to a uint8 tensor
        image = tf.decode_raw(features['image_raw'], tf.float64)

        # Reshape into a 144 x 160 x 3 image and apply distortions
        image = tf.reshape(image, (FLAGS.input_rows, FLAGS.input_cols, FLAGS.input_filters))

        image = data.data_sets.normalize(image)

        return image

    def inputs(self, train, num_epochs):
        if not num_epochs: num_epochs = None
        filename = TRAIN_FILE if train else VALIDATION_FILE

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                            [filename], num_epochs=num_epochs)

            image = self.read_and_decode(filename_queue)
            image = tf.reverse(image, axis=[-1])

            # Shuffle the examples and collect them into batch_size batches.
            images = tf.train.shuffle_batch(
                [image], batch_size=FLAGS.batch_size,
                capacity=1000 + 3 * FLAGS.batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)

            tf.summary.image("input", images, max_outputs=FLAGS.batch_size)

            return images

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, FLAGS.input_filters, FLAGS.C1, "d_h1")) # 144x160x3 -> 72x80x16
            h2 = lrelu(conv2d(h1, FLAGS.C1, FLAGS.C2, "d_h2")) # 72x80x16 -> 36x40x32
            h2_flat = tf.reshape(h2,
                                 [tf.shape(self.images)[0],
                                 (FLAGS.input_rows/4)*(FLAGS.input_cols/4)*FLAGS.C2])

            w_mean = dense(h2_flat, (FLAGS.input_rows/4)*(FLAGS.input_cols/4)*FLAGS.C2, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, (FLAGS.input_rows/4)*(FLAGS.input_cols/4)*FLAGS.C2, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, (FLAGS.input_rows/4)*(FLAGS.input_cols/4)*FLAGS.C2, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [tf.shape(self.images)[0], (FLAGS.input_rows/4), (FLAGS.input_cols/4), FLAGS.C2]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, FLAGS.C2, [tf.shape(self.images)[0], (FLAGS.input_rows/2), (FLAGS.input_cols/2), FLAGS.C1], name="g_h1"))
            h2 = conv_transpose(h1, FLAGS.C1, [tf.shape(self.images)[0], FLAGS.input_rows, FLAGS.input_cols, FLAGS.input_filters], name="g_h2")
            h2 = tf.nn.sigmoid(h2)
            tf.summary.image("generated", h2, max_outputs=FLAGS.batch_size*3)

        return h2

    def train(self):

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Add the variable initializer Op.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Run the Op to initialize the variables.
        sess.run(init_op)
        # Write the graph to TensorBoard
        self.writer.add_graph(sess.graph)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=2)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
          step = 0
          while not coord.should_stop():
            start_time = time.time()

            _, gen_loss, lat_loss = sess.run([self.optimizer, self.generation_loss, self.latent_loss])

            duration = time.time() - start_time

            # Print an overview
            if step % 10 == 0:
              print('Step %d: genloss = %.2f latloss = %.2f (%.3f sec)' % (step, np.mean(gen_loss), np.mean(lat_loss),
                                                                        duration))
              summary_str = sess.run(summary)
              self.writer.add_summary(summary_str, step)
              self.writer.flush()
            step += 1

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 300 == 0:
              saver.save(sess, os.getcwd()+"/training/train",global_step=step)

        except tf.errors.OutOfRangeError:
          print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
          # When done, ask the threads to stop.
          coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()



