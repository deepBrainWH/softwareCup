import tensorflow as tf


def build_model():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 200, 200, 3], "x")
        y = tf.placeholder(tf.float32, [None, 4], "y")

    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(x, 64, [3, 3], activation=tf.nn.relu, name='conv1')
        max1 = tf.layers.max_pooling2d(conv1, [2,2], [2,2])
        bn1 = tf.layers.batch_normalization(max1, name='bn1')
        tf.layers.dropout(bn1, name='droput')

    with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(max1)
