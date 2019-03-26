import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import cv2


class NumModel:
    def __init__(self):
        self.df = pd.read_csv(r"C:\Users\wangheng\Documents\software_cup\dataframe1.csv", index_col=0)
        values = self.df.values
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(values[:, 0], values[:, 2:],
                                                                                test_size=0.3, random_state=0,
                                                                                shuffle=True)


class batch:
    def __init__(self, batch_size=128):
        self.start = 0
        self.offset = batch_size
        self.has_next_batch = True
        model = NumModel()
        self.train_x, self.train_y = model.train_x, model.train_y
        self.test_x, self.test_y = model.test_x, model.test_y
        self.end = self.train_x.shape[0]
        self.batches = math.ceil(self.end / self.offset)

    def get_next_batch(self):
        global one_batch_path_x, one_batch_y
        if self.start + self.offset <= self.end:
            one_batch_path_x = self.train_x[self.start: self.start + self.offset]
            one_batch_y = self.train_y[self.start: self.start + self.offset]
            self.start += self.offset
            if self.start >= self.end:
                self.has_next_batch = False
            else:
                self.has_next_batch = True
        elif self.start + self.offset > self.end:
            one_batch_path_x = self.train_x[self.start:]
            one_batch_y = self.train_y[self.start:]
            self.start = self.end + 1
            self.has_next_batch = False
        train_x = []
        for f in one_batch_path_x:
            imread = cv2.imread(f)
            train_x.append(imread)
        train_x = np.asarray(train_x, np.float)
        train_y = np.asarray(one_batch_y, np.float)
        return train_x, train_y

    def get_test_data(self):
        test_x = []
        for f in self.test_x:
            imread = cv2.imread(f)
            test_x.append(imread)
        test_x = np.asarray(test_x, np.float)
        test_y = np.asarray(self.test_y, np.float)
        return test_x, test_y

    def load_data(self):
        """
        :return: all train data and test data, which include train_x, train_y, test_x, test_y.
        """
        train_x = []
        for f in self.train_x:
            train_x.append(cv2.imread(f))

        test_x = []
        for f in self.test_x:
            test_x.append(cv2.imread(f))
        return np.asarray(train_x, np.float32), np.asarray(self.train_y, np.float32), np.asarray(test_x,
                                                                                             np.float32), np.asarray(
            self.test_y, np.float32)
