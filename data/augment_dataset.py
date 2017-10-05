from data_sets import apply_augmentation
from glob import glob
from random import randint
import cv2
import tensorflow as tf

from flags import FLAGS

# Get all the filenames in the training set
train_names = glob(FLAGS.train_dir + '/*')

def apply_augmentation(image):
    # image augmentation
    augment_type = randint(0, 4)
    if augment_type == 0:
        # random brightness
        print 'random brightness applied'
        image = tf.image.random_brightness(image, max_delta=32./255.)
    elif augment_type == 1:
        # random contrast
        print 'random contrast applied'
        image = tf.image.random_contrast(image, 0.5, 1.5)
    elif augment_type == 2:
        # random flip left right
        print 'random horizontal flip applied'
        image = tf.image.random_flip_left_right(image)
    elif augment_type == 3:
        # random hue
        print 'random hue applied'
        image = tf.image.random_hue(image, 0.05)
    elif augment_type == 4:
        # random saturation
        print 'random saturation applied'
        image = tf.image.random_saturation(image, 0.2, 0.8)
    return image.eval(), augment_type

sess = tf.Session()
with sess.as_default():
    for fn in train_names:
        im = cv2.imread(fn)
        aug_im, aug_type = apply_augmentation(im)
        aug_fn = fn.split('.')[0] + '_' + str(aug_type) + '.png'
        cv2.imwrite(aug_fn, aug_im)
