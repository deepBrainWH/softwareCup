import tensorflow as tf
import cv2


class YoloV1(object):
    num_of_class = 4

    classes = [
        'person',
        'smartphone',
        'bottle',
        'car'
    ]

    def __init__(self, leaky_relu=0.1):
        self.leaky_relu = leaky_relu

    def conv_layer(self, index, inputMatrix, numFilters, sizeOfFilter, stride):
        out = tf.layers.conv2d(inputMatrix, numFilters, sizeOfFilter,
                               (stride, stride), 'same', name='%d_conv' % index)
        out = tf.nn.leaky_relu(out, self.leaky_relu, name='%d_leaky_relu' % index)
        out = tf.layers.max_pooling2d(out, 2, (2, 2), 'same', name='%d_max_pooling' % index)
        return out
