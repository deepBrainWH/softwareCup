import tensorflow as tf
import nn
import dataset

NUM_CLASSES = 11


def model(with_log=True):
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, [None, 41, 28, 3], "input_x")
        y = tf.placeholder(tf.float32, [None, 11], "input_y")
    # conv1 = nn.conv(X, [3, 3, 3, 32], [1, 1], True, name="conv1")
    conv1 = tf.layers.conv2d(X, 32, [3, 3], padding='same', name='conv1')
    tf.layers.batch_normalization(conv1, name="batch_normalization")

    conv2 = nn.conv(conv1, [3, 3, 32, 64], [1, 1], True, name="conv2")
    max2 = nn.max_pooling(conv2, [2,2],[2,2],'SAME')
    flatten = tf.reshape(max2, (-1, max2.shape[1].value*max2.shape[2].value*max2.shape[3].value), name="flatten")
    fc1 = nn.fc(flatten, 128, 'relu', keep_prob=0.7)
    fc2 = nn.fc(fc1, 64, 'relu', keep_prob=0.5)
    fc3 = nn.fc(fc2, NUM_CLASSES, 'softmax')

    predict = fc3
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=predict, name="cost")
    loss = tf.reduce_mean(cost)
    opt = tf.train.AdadeltaOptimizer().minimize(loss)
    correct_predict = tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_predict)

    if with_log:
        print("conv1:\t" + "input-->" + str(X.shape.as_list()[1:]) + "\toutput-->" + str(conv1.shape.as_list()[1:]))
        print("conv2:\t" + "input-->" + str(conv1.shape.as_list()[1:]) + "\toutput-->" + str(conv2.shape.as_list()[1:]))
        print("max2:\t" + "input-->" + str(conv2.shape.as_list()[1:]) + "\toutput-->" + str(max2.shape.as_list()[1:]))
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
    # saver.restore(sess, tf.train.latest_checkpoint("./h5/"))
    sess.run(init)
    for i in range(1000):
        mybatch = dataset.batch()
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
            writer.add_summary(merge_all_, i*batches + j)
        if i % 10 == 0:
            test_x, test_y = mybatch.get_test_data()
            l_, a_ = sess.run([loss, accuracy], feed_dict={X: test_x, y: test_y})
            print("step : %d, test data loss value is : %.4f, test data accuracy is : %.4f" % (i, l_, a_))
        if i % 40 == 0:
            saver.save(sess, "./h5/model.ckpt", i)

    sess.close()

if __name__ == '__main__':
    model()