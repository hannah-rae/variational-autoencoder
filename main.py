import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *

from flags import FLAGS

class LatentAttention():
    def __init__(self):
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = FLAGS.batch_size

        self.images = tf.placeholder(tf.float32, [None, 784], name='input')
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1], name='reshaped_input')
        z_mean, z_stddev = self.recognition(image_matrix)

        # Sample from Gaussian distribution
        samples = tf.random_normal([tf.shape(self.images)[0], self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [tf.shape(self.images)[0], 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-6 + generated_flat) + (1-self.images) * tf.log(1e-6 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, FLAGS.C1, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, FLAGS.C1, FLAGS.C2, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[tf.shape(self.images)[0], 7*7*FLAGS.C2])

            w_mean = dense(h2_flat, 7*7*FLAGS.C2, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*FLAGS.C2, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*FLAGS.C2, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [tf.shape(self.images)[0], 7, 7, FLAGS.C2]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, FLAGS.C2, [tf.shape(self.images)[0], 14, 14, FLAGS.C1], name="g_h1"))
            h2 = conv_transpose(h1, FLAGS.C1, [tf.shape(self.images)[0], 28, 28, 1], name="g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        # Get the image data for one batch as reference in training
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        # Reshape to be batch size x rows x cols
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        # Reshape into a grid and save it as a reference
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.writer.add_graph(sess.graph)
            for epoch in range(FLAGS.num_epochs):
                # Calculate total batches to go through in one epoch as total samples / batchsize
                for idx in range(int(self.n_samples / self.batchsize)):
                    # Get only the image data for one batch
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                # Test on reference images from above after every epoch
                print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                generated_test = generated_test.reshape(self.batchsize,28,28)
                ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()
