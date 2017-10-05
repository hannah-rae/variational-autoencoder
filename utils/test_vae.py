import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import cv2
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *

num_test_images = 1
metafile = './training/train-9.meta'
imagefile = '/Users/hannahrae/data/mnist/apple.jpg'

sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(metafile)
saver.restore(sess, tf.train.latest_checkpoint('./training/'))

graph = tf.get_default_graph()
input_tensor = graph.get_tensor_by_name("input:0")
generated_images = graph.get_tensor_by_name("generation/Sigmoid:0")
cost = graph.get_tensor_by_name("Mean:0")

img = cv2.imread(imagefile, 0)
img = img.reshape(num_test_images, -1)
reconstructed, recon_error = sess.run([generated_images, cost], feed_dict={input_tensor: img})
reconstructed = reconstructed.reshape(num_test_images, 28, 28)
ims("results/exp0_1/apple_" + str(recon_error) + ".jpg", reconstructed[0])