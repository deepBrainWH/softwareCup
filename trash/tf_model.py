import tensorflow as tf


def build_model():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 200, 200, 3], "x")
        y = tf.placeholder(tf.float32, [None, 6], "y")

    with tf.variable_scope("conv_layer_1"):
        conv1 = tf.layers.conv2d(x, 64, [3, 3], activation=tf.nn.relu, name='conv1')
        max1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
        bn1 = tf.layers.batch_normalization(max1, name='bn1')
        output1 = tf.layers.dropout(bn1, name='droput')

    with tf.variable_scope("conv_layer_2"):
        conv2 = tf.layers.conv2d(output1, 64, [3, 3], activation=tf.nn.relu, name='conv2')
        max2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2], name='max2')
        bn2 = tf.layers.batch_normalization(max2)
        output2 = tf.layers.dropout(bn2, name='dropout')

    with tf.variable_scope("conv_layer_3"):
        conv3 = tf.layers.conv2d(output2, 64, [3, 3], activation=tf.nn.relu, name='conv3')
        max3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2], name='max3')
        bn3 = tf.layers.batch_normalization(max3, name='bn3')
        output3 = bn3

    with tf.variable_scope("conv_layer_4"):
        conv4 = tf.layers.conv2d(output3, 32, [3, 3], activation=tf.nn.relu, name='conv4')
        max4 = tf.layers.max_pooling2d(conv4, [2, 2], [2, 2], name='max4')
        bn4 = tf.layers.batch_normalization(max4, name='bn4')
        output = bn4
        flatten = tf.layers.flatten(output, 'flatten')

    with tf.variable_scope("fc_layer1"):
        fc1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
        fc_bn1 = tf.layers.batch_normalization(fc1, name='bn1')
        dropout1 = tf.layers.dropout(fc_bn1, 0.5)

    with tf.variable_scope("fc_layer2"):
        fc2 = tf.layers.dense(dropout1, 128, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(fc2)

    with tf.variable_scope("fc_layer3"):
        fc3 = tf.layers.dense(dropout2, 64)
        dropout3 = tf.layers.dropout(fc3)

    with tf.variable_scope("fc_layer4"):
        fc4 = tf.layers.dense(dropout3, 32)

    with tf.variable_scope("fc_layer5"):
        fc5 = tf.layers.dense(fc4, 6)

    softmax = tf.nn.softmax(fc5, name='softmax')
    predict = tf.argmax(softmax, axis=1)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc5, labels=y, name='loss'))
    tf.summary.scalar("loss", loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(y, axis=1)), tf.float32))
    tf.summary.scalar("acc", accuracy)
    merged = tf.summary.merge_all()
    return x, y, predict, loss, accuracy, merged
