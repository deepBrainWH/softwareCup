import tensorflow as tf


def conv(X, kernel_shape, stride, with_bn=True, activation='relu', use_bias=True, padding='SAME', device='/GPU:0',
         name='ConvNet'):
    """
    :param X: input tensor: 4-dim: [batch_size, height, width, channel]
    :param kernel_shape: filter kernel: 4-dim: [x, y, in-dim, out-dim]
    :param stride: 2-dim: [x, y]
    :param with_bn: if this layer use batch normalization.
    :param activation: activation function
    :param use_bias: add bias
    :param padding: use padding: SAME or VALID
    :param name: layers name
    :param with_pooling: default is max_pool.
    :param pooling_size: 2-dim. if with_pooling is true, you have to assign the pooling size.
    :param pooling_stride:  2-dim. if with_pooling is true, you have to assign the pooling stride.
    :param pooling_padding: default value is 'SAME', you can also use 'VALID'.
    :return: convolution result and result shape.
    """
    with tf.name_scope(name):
        with tf.device(device):
            kernel = tf.Variable(tf.truncated_normal(kernel_shape, 0.0, 0.1), name='kernel')
            conv_ = tf.nn.conv2d(X, kernel, [1, stride[0], stride[1], 1], padding, name='conv_layer')
            if use_bias:
                bias_ = tf.Variable(tf.zeros([kernel_shape[3]], name='bias'))
                conv_ = tf.nn.bias_add(conv_, bias_, name='bias_add')
            if activation == 'relu':
                conv_ = tf.nn.relu(conv_, activation)
            elif activation == 'relu6':
                conv_ = tf.nn.relu6(conv_, activation)
            elif activation == 'sigmoid':
                conv_ = tf.nn.sigmoid(conv_, activation)
            elif activation == 'tanh':
                conv_ = tf.nn.tanh(conv_, activation)
            elif activation == 'leaky_relu':
                conv_ = tf.nn.leaky_relu(conv_, 0.2, activation)

            if with_bn:
                means_, vars_ = tf.nn.moments(conv_, [0, 1, 2])
                offset = tf.zeros([1, kernel_shape[3]])
                scale = tf.ones([1, kernel_shape[3]])
                epsilon = tf.ones([1, kernel_shape[3]]) * 1e-6
                normalization = tf.nn.batch_normalization(conv_, means_, vars_, offset, scale, epsilon,
                                                          "batch_normalization")
                return normalization
            return conv_


def max_pooling(X, kernel_shape, stride, padding='SAME', name='max_pooling'):
    """
    :param X: input value.
    :param kernel_shape: kernel shape, 2 dimension.
    :param stride: 2 dimension
    :param padding: SAME or VALID
    :param name:
    :return:
    """
    if len(kernel_shape) != 2: raise Exception("the kernel shape of max pooling must be 2 dimension")
    return tf.nn.max_pool(X, [1, kernel_shape[0], kernel_shape[1], 1], [1, stride[0], stride[1], 1], padding, name=name)


def fc(X, unit_size, activation='sigmoid', device='/GPU:0', name='full_connect', keep_prob=1.0):
    """
    :param X: input tensor
    :param unit_size: the unit size of this layer.
    :param activation: activation function. you can use: 'relu', 'sigmoid', 'softmax', 'relu6', 'tanh'
    :param name: layer's name.
    :return:
    """
    with tf.name_scope(name):
        with tf.device(device):
            if len(X.get_shape()) != 2:
                raise Exception("The input tensor 'X' must have 2 dimension")
            in_dim = X.get_shape()[1].value
            w = tf.Variable(tf.random.normal([in_dim, unit_size], name='weight'))
            b = tf.Variable(tf.random.normal([unit_size], name='bias'))
            fc_ = tf.matmul(X, w, name='matmul')
            fc_ = tf.nn.bias_add(fc_, b)
            if activation == 'sigmoid':
                fc_ = tf.nn.sigmoid(fc_, activation)
            elif activation == 'relu':
                fc_ = tf.nn.relu(fc_, activation)
            elif activation == 'relu6':
                fc_ = tf.nn.relu6(fc_, activation)
            elif activation == 'tanh':
                fc_ = tf.nn.tanh(fc_, activation)
            elif activation == 'softmax':
                fc_ = tf.nn.softmax(fc_, name=activation)
            fc_ = tf.nn.dropout(fc_, keep_prob, name='dropout')
            return fc_
