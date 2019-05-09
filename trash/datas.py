import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import tensorflow as tf

class Data(object):
    def __init__(self, batch_size=64):
        self.image_path = "C:\\Users\\wangheng\\Documents\\software_cup\\trash\\"
        self.start = 0
        self.end = len(os.listdir(self.image_path))
        self.has_next_batch = True
        self.batch_size = batch_size

    def rename(self):
        listdir = os.listdir(self.image_path)
        i = 0
        while i < len(listdir):
            os.rename(os.path.join(self.image_path, listdir[i]), os.path.join(self.image_path, "0_%d" % i + ".jpg"))
            i += 1

    def resize_img(self):
        listdir = os.listdir(self.image_path)
        for file in listdir:
            imread = cv2.imread(os.path.join(self.image_path, file))
            resize = cv2.resize(imread, (200, 200))
            cv2.imwrite(os.path.join(self.image_path, file), resize)

    def load_all_data(self):
        files = os.listdir(self.image_path)
        img = []
        label = []
        for file in files:
            lab = np.zeros((4))
            os.path.join(self.image_path, file)
            image = np.asarray(image, np.float32) / 255.
            if file[0] == '0':
                lab[0] = 1
            elif file[0] == '1':
                lab[1] = 1
            elif file[0] == '2':
                lab[2] = 1
            elif file[0] == '3':
                lab[3] = 1

            img.append(image)
            label.append(lab)
        train_x = np.asarray(img, np.float32)
        train_y = np.asarray(label, np.float32)
        np.random.seed(10)
        np.random.shuffle(train_x)
        np.random.shuffle(train_y)
        return train_x[0:1000], train_y[0:1000]

    def train_data_to_csv(self):
        files = os.listdir(self.image_path)
        data = []
        for file in files:
            data.append({"path": self.image_path + file, "label": file[0]})

        frame = pd.DataFrame(data, columns=['path', 'label'])
        dummies = pd.get_dummies(frame['label'], 'label')
        concat = pd.concat([frame, dummies], 1)
        concat.to_csv(self.image_path + "train.csv")

    def build_model(self):
        train_x, train_y = self.load_all_data()
        model = Sequential()
        model.add(Conv2D(64, [3,3], activation='relu', input_shape=(200, 200, 3)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, [5,5], activation='relu'))
        model.add(MaxPooling2D([2,2],[2,2]))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, [3,3], activation='relu'))
        model.add(Conv2D(128, [3,3], activation='relu'))
        model.add(MaxPooling2D([2,2],[2,2]))
        model.add(Conv2D(64, [3,3], activation='relu'))
        model.add(MaxPooling2D([2,2],[2,2]))
        model.add(Conv2D(32, [3,3], [2,2], activation='relu'))
        model.add(MaxPooling2D([2,2], [2,2]))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=Adadelta(), loss=categorical_crossentropy, metrics=['accuracy'])
        print(model.summary())

        filepath = "./h5_deep_server/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5_deep_server"
        callback_list = [TensorBoard('./log'),
                         ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)]
        model.fit(train_x, train_y, 64, 10, 1, callbacks=callback_list, validation_split=0.3,shuffle=True)

if __name__ == '__main__':
    obj = Data()
    # obj.rename()
    # obj.resize_img()
    # train_x, train_y = obj.load_all_data()
    # print(train_x.shape, train_y.shape)
    # obj.build_model()
    obj.train_data_to_csv()