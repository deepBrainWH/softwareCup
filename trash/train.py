import tf_model
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import cv2
import os

# np.random.seed(10)


class Train(object):
    def __init__(self, data_path, batch_size=128):
        self.batch_size = batch_size
        self.start = 0
        self.data_path = data_path
        self.train_x_path, self.train_y, self.end, self.test_x_path, self.test_y = self.__get_train_data()
        self.has_next_batch = True
        self.batches = math.ceil(self.end / self.batch_size)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.max_train_accuracy = -1
        self.max_test_accuracy = -1

        # Following codes are the corresponding parameters of model.
        self.x, self.y, self.predict, self.loss, self.accuracy, self.merged, self.softmax = tf_model.build_model()
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def __get_train_data(self, train_test_split=0.9):
        frame = pd.read_csv(self.data_path, index_col=0)
        values = frame.values
        end = math.ceil(values.shape[0] * train_test_split)
        np.random.shuffle(values)
        train_x_path = values[:end, 0]
        test_x_path = values[end:, 0]
        train_y = values[:end, 2:]
        test_y = values[end:, 2:]
        return train_x_path, train_y, train_x_path.shape[0], test_x_path, test_y

    def next_batch(self):
        if self.start + self.batch_size <= self.end:
            batch_x_train_path = self.train_x_path[self.start: self.start + self.batch_size]
            train_y = self.train_y[self.start: self.start + self.batch_size]
            train_x = []
            for f_ in batch_x_train_path:
                img = cv2.imread(f_)
                train_x.append(img)
            train_x = np.asarray(train_x, np.float32) / 255.
            train_y = np.asarray(train_y, np.float32)
            self.start += self.batch_size
            if self.start <= self.end:
                self.has_next_batch = True
            else:
                self.has_next_batch = False
            return train_x, train_y
        else:
            batch_x_train_path = self.train_x_path[self.start:]
            train_y = self.train_y[self.start:]
            train_x = []
            for f_ in batch_x_train_path:
                img = cv2.imread(f_)
                train_x.append(img)
            train_x = np.asarray(train_x, np.float32) / 255.
            train_y = np.asarray(train_y, np.float32)
            self.start = self.end
            self.has_next_batch = False
            return train_x, train_y

    def get_test_data(self):
        test_x = []
        test_y = self.test_y
        for file in self.test_x_path:
            img = cv2.imread(file)
            test_x.append(img)
        test_x = np.asarray(test_x, np.float32) / 255.
        test_y = np.asarray(test_y, np.float32)
        return test_x, test_y

    def train(self):
        saver = tf.train.Saver(max_to_keep=15)
        if os.path.isdir("./model_pb"):
            os.removedirs("./model_pb")
        # builder = tf.saved_model.builder.SavedModelBuilder("./model_pb/")
        sess = tf.InteractiveSession(config=self.config)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./log", graph=sess.graph)
        for i in range(100):
            j = 1
            all_loss_ = 0
            all_acc_ = 0
            while j <= self.batches and self.has_next_batch:
                train_x, train_y = self.next_batch()
                _, loss_, accuracy_, merged_ = sess.run([self.opt, self.loss, self.accuracy, self.merged],
                                                        feed_dict={self.x: train_x, self.y: train_y})
                all_loss_ += loss_
                all_acc_ += accuracy_
                print("\r epoch %d-- batch: %d-->    " % (i, j) + "=" * j + ">" + "-" * (
                        self.batches - j) + "\t\t loss: %.4f, acc: %.4f" % (
                          loss_, accuracy_), end='')
                j += 1
                writer.add_summary(merged_, i * self.batches + j - 1)
            print("\n===epoch %d===    >    mean loss is : %.5f, mean acc is : %.4f" % (
                i, all_loss_ / self.batches, all_acc_ / self.batches))
            test_x, test_y = self.get_test_data()
            all_test_data_size = len(test_y)
            all_test_losses = []
            all_test_acc = []
            all_test_data_batches = math.ceil(all_test_data_size / self.batch_size)
            for k in range(all_test_data_batches):
                if k * self.batch_size + self.batch_size <= all_test_data_size:
                    __start = k * self.batch_size
                    __end = k * self.batch_size + self.batch_size
                else:
                    __start = k * self.batch_size
                    __end = all_test_data_size
                test_loss_, test_acc_ = sess.run([self.loss, self.accuracy],
                                                 feed_dict={self.x: test_x[__start:__end],
                                                            self.y: test_y[__start:__end]})
                all_test_losses.append(test_loss_)
                all_test_acc.append(test_acc_)
            print("===epoch %d===    >    test loss is : %.4f, test acc is : %.4f" % (
                i, np.mean(np.array(all_test_losses)), np.mean(np.array(all_test_acc))))
            self.start = 0
            self.has_next_batch = True
            if all_acc_ / self.batches > self.max_train_accuracy or \
                np.mean(np.array(all_test_acc)) > self.max_test_accuracy:
                saver.save(sess, "./h5_dell/mode.ckpt", i)
                self.max_train_accuracy = all_acc_ / self.batches
                self.max_test_accuracy = np.mean(np.array(all_test_acc))
        sess.close()

    def predict_value(self, type='image', image_path=None):
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        saver.restore(sess, tf.train.latest_checkpoint("./h5_dell/"))
        if type == 'image':
            assert image_path is not None
            image = cv2.imread(image_path)
            if image.shape != (200, 200):
                image = cv2.resize(image, (200, 200))
            image = np.asarray(image, np.float32) / 255.
            image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
            [predict, probabilistic] = sess.run([self.predict, self.softmax], feed_dict={self.x: image})
            probabilistic = np.max(probabilistic)
            print("category: %d, prob: %.4f" % (predict, probabilistic))
        elif type == 'video':
            capture = cv2.VideoCapture(0)
            while True:
                ret, frame = capture.read()
                resize = cv2.resize(frame, (200, 200))
                x_ = np.asarray(resize, np.float32) / 255.
                x_ = np.reshape(x_, [1, x_.shape[0], x_.shape[1], x_.shape[2]])
                [predict, probabilistic] = sess.run([self.predict, self.softmax], feed_dict={self.x: x_})
                probabilistic = np.max(probabilistic)
                if predict == 0:
                    cv2.putText(frame, "0 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)
                elif predict == 1:
                    cv2.putText(frame, "1 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)
                elif predict == 2:
                    cv2.putText(frame, "2 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                elif predict == 3:
                    cv2.putText(frame, "3 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 255), 2, cv2.LINE_AA)
                elif predict == 4:
                    cv2.putText(frame, "4 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("recognized", frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cv2.destroyAllWindows()
            capture.release()

    def trian_model(self):
        while self.has_next_batch:
            train_x, train_y = self.next_batch()
            print(train_x.shape, train_y.shape)
            self.train()
