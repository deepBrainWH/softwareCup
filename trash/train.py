import tf_model
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
np.random.seed(10)


class Train(object):
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.start = 0
        self.data_path = "C:\\Users\\wangheng\\Documents\\software_cup\\train.csv"
        self.train_x_path, self.train_y, self.end = self.__get_train_data()
        self.has_next_batch = True

    def __get_train_data(self):
        frame = pd.read_csv(self.data_path)
        values = frame.values
        np.random.shuffle(values)
        train_x_path = values[:, 0]
        train_y = values[:, 2:]
        return train_x_path, train_y, train_x_path.shape[0]

    def next_batch(self):
        if self.start + self.batch_size <= self.end:
            batch_x_train_path = self.train_x_path[self.start: self.start+self.batch_size]
            train_y = self.train_y[self.start: self.start + self.batch_size]
            train_x = []
            for f_ in batch_x_train_path:
                img = cv2.imread(f_)
                train_x.append(img)
            train_x = np.asarray(train_x, np.float32) / 255.
            train_y = np.asarray(train_y, np.float32)
            self.start += self.batch_size
        

    def train(self, train_x, train_y):
        x, y, predict, loss, accuracy, merged = tf_model.build_model()
        opt = tf.train.AdamOptimizer().minimize(loss)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("./log", graph=sess.graph)

        sess.close()


if __name__ == '__main__':
    mytrain = Train(128)
    # mytrain.
