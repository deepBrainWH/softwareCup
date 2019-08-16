from __future__ import absolute_import, print_function
import os
import cv2
import pandas as pd
from . train import Train

csv_file_saved_path = "/home/wangheng/Documents/software_cup/"

class Data(object):
    def __init__(self, batch_size=64):
        self.image_path = "/home/wangheng/Documents/software_cup/trash/"
        self.start = 0
        self.end = len(os.listdir(self.image_path))
        self.has_next_batch = True
        self.batch_size = batch_size

    def rename(self):
        listdir = os.listdir(self.image_path)
        i = 0
        while i < len(listdir):
            images_list_dir = os.listdir(os.path.join(self.image_path, listdir[i]))
            j = 0
            while j < len(images_list_dir):
                old_name = os.path.join(self.image_path, listdir[i], images_list_dir[j])
                new_name = os.path.join(self.image_path, "%d-%d" % (i, j) + ".jpg")
                os.rename(old_name, new_name)
                j += 1
            i += 1
        for p in range(len(listdir)):
            tmp_path = os.path.join(self.image_path, listdir[p])
            if os.path.exists(tmp_path):
                os.removedirs(tmp_path)

    def resize_img(self):
        listdir = os.listdir(self.image_path)
        for file in listdir:
            file_path = os.path.join(self.image_path, file)
            try:
                imread = cv2.imread(file_path)
                resize = cv2.resize(imread, (200, 200))
                cv2.imwrite(os.path.join(self.image_path, file), resize)
            except Exception:
                os.remove(file_path)
                continue

    def train_data_to_csv(self):
        files = os.listdir(self.image_path)
        data = []
        for file in files:
            data.append({"path": self.image_path + file, "label": file[0]})

        frame = pd.DataFrame(data, columns=['path', 'label'])
        dummies = pd.get_dummies(frame['label'], 'label')
        concat = pd.concat([frame, dummies], 1)
        concat.to_csv(csv_file_saved_path + "train.csv")

if __name__ == '__main__':
    obj = Data()
    obj.rename()
    obj.resize_img()
    obj.train_data_to_csv()

    train_obj = Train(csv_file_saved_path + "train.csv", 32)
    train_obj.train()

    # predict_value = train_obj.predict_value("image", "your image path.")

