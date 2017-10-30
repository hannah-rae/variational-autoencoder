import json
import os
import numpy as np
import tensorflow as tf

from flags import FLAGS

# standard convolution layer
def conv2d(x, kernel_size, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        print inputFeatures
        w = tf.get_variable("w",[kernel_size, kernel_size, inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        return conv

def conv_transpose(x, kernel_size, prev_size, outputShape, name):
    with tf.variable_scope(name):
        # h, w, out, in
        w = tf.get_variable("w",[kernel_size, kernel_size, outputShape[-1], prev_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=tf.stack(outputShape), strides=[1,2,2,1])
        return convt

# def max_pool(x, name):
#     with tf.variable_scope(name):
#         return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

# def upsample(x, name):
#     with tf.variable_scope(name):
#         return tf.contrib.keras.layers.UpSampling2D((2, 2))(x)

# def deconv2d(input_, output_shape,
#              k_h=FLAGS.kernel_size, k_w=FLAGS.kernel_size, d_h=2, d_w=2, stddev=0.02,
#              name="deconv2d"):
#     with tf.variable_scope(name):
#         # filter : [height, width, output_channels, in_channels]
#         w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
#                             initializer=tf.random_normal_initializer(stddev=stddev))

#         deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

#         biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#         deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

#         return deconv

# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias
