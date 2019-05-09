import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import copy
import myconfig


class LoadData(object):
    def __init__(self, path, batch, CLASS):
        self.devkil_path = path + '/VOCdevkit'
        self.image_path = myconfig.credit_card_dir
        self.image_label_path = myconfig.credit_card_label_dir
        self.image_width = 520
        self.image_height = 390
        self.image_size = 416
        self.batch = batch
        self.CLASS = CLASS
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS, range(self.n_class)))

    def load_img(self, PATH):
        im = cv2.imread(PATH)
        im = cv2.resize(im, (self.image_size, self.image_size))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.multiply(1. / 255, im)
        return im

    def load_xml(self, index):
        image_path = self.image_path + str(index) + ".jpg"
        xml_path = self.image_label_path + str(index) + '.xml'
        img = cv2.imread(image_path)
        w = self.image_size / img.shape[0]
        h = self.image_size / img.shape[1]
        # yolo output: batch, 7, 7, 25, 25 = 5 + 20, 5:(x, y, w, h, c), 20:20个类别
        #
        label = np.zeros((7, 7, 7)) # 5 + 2, 5:(x,y,w,h,c), 2: two classes.
        tree = ET.parse(xml_path)
        objs = tree.findall("object")
        for i in objs:
            box = i.find("bndbox")
            x1 = max(min((float(box.find('xmin').text) - 1) * w, self.image_size - 1), 0)
            y1 = max(min((float(box.find('ymin').text) - 1) * h, self.image_size - 1), 0)
            x2 = max(min((float(box.find('xmax').text) - 1) * w, self.image_size - 1), 0)
            y2 = max(min((float(box.find('ymax').text) - 1) * w, self.image_size - 1), 0)
            boxes = [(x1 + x2) / 2., (y1 + y2) / 2., x2 - x1, y2 - y1]
            class_id = self.class_id[i.find("name").text.lower().strip()]

            x_id=int(boxes[0] / self.image_size * 7)
            y_id = int(boxes[1] / self.image_size *7)
            if label[y_id, x_id, 0] == 1:
                continue
            label[y_id, x_id, 0] = 1 # confident
            label[y_id, x_id, 1:5] = boxes
            label[y_id, x_id, 5+class_id] = 1
        return label, len(objs)

    def load_label(self):
        pass



if __name__ == "__main__":
    CLASSESE = ['card', 'number']
    data_path = '../data/pascal_voc'
    test = LoadData(data_path, 10, CLASSESE)
    print(test.class_id['number'])
