from __future__ import absolute_import, print_function
import sys

sys.path.append("..")
import os
import cv2
import pandas as pd

csv_file_saved_path = "./training_data/"
image_path = "./training_data/new_datasets/"


class Data(object):
    def __init__(self, batch_size=64):
        self.start = 0
        try:
            self.end = len(os.listdir(image_path))
        except:
            print("Image dir not exit: `%s`" % image_path)
            exit(-1)
        self.has_next_batch = True
        self.batch_size = batch_size

    def rename(self):
        listdir = os.listdir(image_path)
        listdir.sort()
        i = 0
        while i < len(listdir):
            images_list_dir = os.listdir(os.path.join(image_path, listdir[i]))
            j = 0
            images_list_dir.sort()
            while j < len(images_list_dir):
                old_name = os.path.join(image_path, listdir[i], images_list_dir[j])
                new_name = os.path.join(image_path, "%d-%d" % (i, j) + ".jpg")
                os.rename(old_name, new_name)
                j += 1
            i += 1
        for p in listdir:
            tmp_path = os.path.join(image_path, p)
            if os.path.exists(tmp_path):
                os.removedirs(tmp_path)

    def resize_img(self):
        listdir = os.listdir(image_path)
        for file in listdir:
            file_path = os.path.join(image_path, file)
            try:
                imread = cv2.imread(file_path)
                resize = cv2.resize(imread, (200, 200))
                cv2.imwrite(os.path.join(image_path, file), resize)
            except Exception:
                os.remove(file_path)
                continue

    def train_data_to_csv(self):
        files = os.listdir(image_path)
        data = []
        for file in files:
            data.append({"path": image_path + file, "label": file[0]})
        frame = pd.DataFrame(data, columns=['path', 'label'])
        dummies = pd.get_dummies(frame['label'], 'label')
        concat = pd.concat([frame, dummies], 1)
        concat.to_csv(csv_file_saved_path + "train.csv")


if __name__ == '__main__':
    obj = Data()
    # obj.rename()
    # obj.resize_img()
    obj.train_data_to_csv()
