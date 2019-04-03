import tf_model
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import cv2

np.random.seed(10)


class Train(object):
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.start = 0
        self.data_path = "C:\\Users\\wangheng\\Documents\\software_cup\\train.csv"
        self.train_x_path, self.train_y, self.end, self.test_x_path, self.test_y = self.__get_train_data()
        self.has_next_batch = True
        self.batches = math.ceil(self.end / self.batch_size)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # 一下为模型相关参数
        self.x, self.y, self.predict, self.loss, self.accuracy, self.merged = tf_model.build_model()
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def __get_train_data(self, train_test_split=0.3):
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

        saver = tf.train.Saver(max_to_keep=3)
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
                print("\repoch %d-- batch: %d-->    " % (i, j) + "=" * j + ">" + "-" * (self.batches - j) + "\t\t loss: %.4f, acc: %.4f" % (
                    loss_, accuracy_), end='')
                j += 1
                writer.add_summary(merged_, i * self.batches + j - 1)
            print("\n===epoch %d===    >    mean loss is : %.4f, mean acc is : %.4f" % (
                i, all_loss_ / self.batches, all_acc_ / self.batches))
            test_x, test_y = self.get_test_data()
            test_loss_, test_acc_ = sess.run([self.loss, self.accuracy], feed_dict={self.x: test_x[0:16], self.y: test_y[0:16]})
            print("===epoch %d===    >    test loss is : %.4f, test acc is : %.4f" % (
                i, test_loss_, test_acc_))
            self.start = 0
            self.has_next_batch = True
            if i % 5 == 0:
                saver.save(sess, "./h5_dell/mode.ckpt", i)
        sess.close()

    def predict_value(self, type='image', image_path=None):
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        saver.restore(sess, tf.train.latest_checkpoint("./h5_dell/"))
        if type == 'image':
            image = cv2.imread(image_path)
            image = np.asarray(image, np.float32) / 255.
            image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
            predict = sess.run(self.predict, feed_dict={self.x: image})
        elif type == 'video':
            capture = cv2.VideoCapture(1)
            while True:
                ret, frame = capture.read()
                resize = cv2.resize(frame, (200, 200))
                x_ = np.asarray(resize, np.float32) / 255.
                x_ = np.reshape(x_, [1, x_.shape[0], x_.shape[1], x_.shape[2]])
                predict = sess.run(self.predict, feed_dict={self.x: x_})
                if predict == 0:
                    cv2.putText(frame, "bottle", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                elif predict == 1:
                    cv2.putText(frame, "paper", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
                elif predict == 2:
                    cv2.putText(frame, "fruit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                elif predict == 3:
                    cv2.putText(frame, "food", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("recognized", frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cv2.destroyAllWindows()
            capture.release()


if __name__ == '__main__':
    mytrain = Train(32)
    # while mytrain.has_next_batch:
    #     train_x, train_y = mytrain.next_batch()
    #     print(train_x.shape, train_y.shape)
    # mytrain.train()
    mytrain.predict_value("video")
