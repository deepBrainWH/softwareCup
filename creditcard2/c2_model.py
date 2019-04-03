import tensorflow as tf
import c2_dataset

NUM_CLASSES = 11


def model(with_log=True):
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, [None, 41, 28, 3], "input_x")
        y = tf.placeholder(tf.float32, [None, 11], "input_y")
    conv1 = tf.layers.conv2d(inputs=X,
                             filters=32,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             # kernel_regularizer=regularizer,
                             name='conv1')
    bn1 = tf.layers.batch_normalization(conv1, name="bn1")
    dropout1 = tf.layers.dropout(bn1, 0.5, name="dropout1")
    conv2 = tf.layers.conv2d(inputs=dropout1,
                             filters=64,
                             kernel_size=[3, 3],
                             activation=tf.nn.relu,
                             # kernel_regularizer=regularizer,
                             name="conv2")
    bn2 = tf.layers.batch_normalization(conv2, name='bn2')
    max1 = tf.layers.max_pooling2d(bn2, [2, 2], [2, 2], 'same')
    dropout2 = tf.layers.dropout(max1, 0.5, name='dropout2')
    flatten = tf.reshape(dropout2, (-1, dropout2.shape[1].value * dropout2.shape[2].value * dropout2.shape[3].value),
                         name="flatten")

    fc1 = tf.layers.dense(flatten, 128, activation=tf.nn.relu)
    bn3 = tf.layers.batch_normalization(fc1, name="bn3")
    dropout3 = tf.layers.dropout(bn3, 0.5)
    fc2 = tf.layers.dense(dropout3, 64, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(fc2, 0.5)
    fc3 = tf.layers.dense(dropout4, NUM_CLASSES)

    predict = tf.nn.softmax(fc3)

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc3, name="cost")
    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.reduce_mean(cost) + l2_loss
    opt = tf.train.AdadeltaOptimizer().minimize(loss)
    correct_predict = tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_predict)

    if with_log:
        print("conv1:\t" + "input-->" + str(X.shape.as_list()[1:]) + "\toutput-->" + str(conv1.shape.as_list()[1:]))
        print("conv2:\t" + "input-->" + str(conv1.shape.as_list()[1:]) + "\toutput-->" + str(conv2.shape.as_list()[1:]))
        print("max2:\t" + "input-->" + str(conv2.shape.as_list()[1:]) + "\toutput-->" + str(max1.shape.as_list()[1:]))
        print("fc1:\t" + "input-->" + str(flatten.shape.as_list()[1:]) + "\toutput-->" + str(fc1.shape.as_list()[1:]))
        print("fc2:\t" + "input-->" + str(fc1.shape.as_list()[1:]) + "\toutput-->" + str(fc2.shape.as_list()[1:]))
        print("fc3:\t" + "input-->" + str(fc2.shape.as_list()[1:]) + "\toutput-->" + str(fc3.shape.as_list()[1:]))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("acc", accuracy)

    merge_all = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter('./log', sess.graph)
    saver = tf.train.Saver(max_to_keep=3)
    # saver.restore(sess, tf.train.latest_checkpoint("./h5_deep_server/"))
    sess.run(init)
    for i in range(1000):
        mybatch = c2_dataset.batch()
        batches = mybatch.batches
        tmp_loss_ = 0
        tmp_acc_ = 0
        j = 0
        while mybatch.has_next_batch:
            x_, y_ = mybatch.get_next_batch()
            _, loss_, acc_, merge_all_ = sess.run([opt, loss, accuracy, merge_all], feed_dict={X: x_, y: y_})
            tmp_loss_ += loss_
            tmp_acc_ += acc_
            j += 1
            writer.add_summary(merge_all_, i * batches + j)
        print("step %d , loss is : %.4f, acc is : %.4f" % (i, tmp_loss_ / batches, tmp_acc_ / batches))
        if i % 10 == 0:
            test_x, test_y = mybatch.get_test_data()
            l_, a_ = sess.run([loss, accuracy], feed_dict={X: test_x / 255.0, y: test_y})
            print("step : %d, test data loss value is : %.4f, test data accuracy is : %.4f" % (i, l_, a_))
        if i % 40 == 0:
            saver.save(sess, "./h5_deep_server/model.ckpt", i)

    sess.close()


if __name__ == '__main__':
    model()
