from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from random import shuffle, randint
from utils.flags import FLAGS

class DataSet(object):

    def __init__(self, input_rows, input_cols, num_filters):
        self.input_cols = input_cols
        self.input_rows = input_rows
        self.num_filters = num_filters

        self.train = self.load_training_data()
        self.test = self.load_test_data()
        self.validation = self.load_validation_data()

    def load_data(self, d):
        image_list = glob(d  + '/*')
        images = np.ndarray((len(image_list), self.input_rows, self.input_cols, self.num_filters))
        for i, img_fn in enumerate(image_list):
            #images[i] = np.divide(cv2.imread(img_fn), 256.)
            images[i] = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE).reshape((self.input_rows, self.input_cols, self.num_filters))
        return images

    def load_training_data(self):
        return self.load_data(FLAGS.train_dir)

    def load_test_data(self):
        return self.load_data(FLAGS.test_dir)

    def load_validation_data(self):
        return self.load_data(FLAGS.val_dir)


def normalize(image):
    # subtract mean, divide by standard deviation
    # https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    return tf.image.per_image_standardization(image)

def apply_augmentation(image):
    # image augmentation
    augment_type = randint(0, 4)
    if augment_type == 0:
        # random brightness
        print 'random brightness applied'
        image = tf.image.random_brightness(image, 63)
    elif augment_type == 1:
        # random contrast
        print 'random contrast applied'
        image = tf.image.random_contrast(image, 0.2, 1.8)
    elif augment_type == 2:
        # random flip left right
        print 'random horizontal flip applied'
        image = tf.image.random_flip_left_right(image)
    elif augment_type == 3:
        # random hue
        print 'random hue applied'
        image = tf.image.random_hue(image, 0.1)
    elif augment_type == 4:
        # random saturation
        print 'random saturation applied'
        image = tf.image.random_saturation(image, 0.1, 0.5)
    return image, augment_type

if __name__ == '__main__':
    test()