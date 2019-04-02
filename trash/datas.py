import os
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization


class Data(object):
    def __init__(self):
        self.image_path = "C:\\Users\\wangheng\\Documents\\software_cup\\trash\\饭盒\\"

    def rename(self):
        listdir = os.listdir(self.image_path)
        i = 0
        while i < len(listdir):
            os.rename(self.image_path + listdir[i], self.image_path + "3_%d" % i + ".jpg")


if __name__ == '__main__':
    obj = Data()
    obj.rename()
