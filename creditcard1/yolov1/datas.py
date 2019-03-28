import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import copy


class LoadData(object):
    def __init__(self, path, batch, CLASS):
        self.devkil_path = path + '/VOCdevkit'
        self.data_path = self.devkil_path + "/VOC2007"
        self.image_width = 520
        self.image_height = 390
        self.batch = batch
        self.CLASS = CLASS
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS, range(self.n_class)))

    def load_img(self, PATH):
        im = cv2.imread(PATH)
        im = cv2.resize(im, (self.image_width, self.image_height))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.multiply(1. / 255, im)
        return im

    def load_xml(self, index):
        path = self.data_path + "/images/" + index + ".jpg"
        xml_path = self.data_path +"/annotations/" + index + '.xml'
        img = cv2.imread(path)
        w = self.image_width / img.shape[0]
        h = self.image_height / img.shape[1]
        # yolo output: batch, 7, 7, 25, 25 = 5 + 20, 5:(x, y, w, h, c), 20:20个类别
        #
        label = np.zeros((7, 7, 10))
        tree = ET.parse(xml_path)
        objs = tree.findall("object")
        for i in objs:
            pass
            


if __name__ == "__main__":
    CLASSESE = ['card', 'number', 'others']
    data_path = '../data/pascal_voc'
    test = LoadData(data_path, 10, CLASSESE)
    print(test.class_id)
