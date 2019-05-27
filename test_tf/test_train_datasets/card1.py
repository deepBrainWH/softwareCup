import cv2
import numpy as np
import os
from test_tf.test_train_datasets.utils import *


FILE_PATH = "/home/wangheng/Documents/software_cup/train_card_images/"
listdir = os.listdir(FILE_PATH)
files = []
for file in listdir:
    res = os.path.join(FILE_PATH, file)
    if os.path.isfile(res):
        files.append(res)

def read_image(image_path):
    imread = cv2.imread(image_path)

    np_max = np.max(imread, axis=2)
    np_asarray1 = np.asarray(np_max, dtype=np.uint8)

    np_min = np.min(imread, axis=2)
    np_asarray2 = np.asarray(np_min, dtype=np.uint8)

    my_gray = cv2.cvtColor(imread, cv2.COLOR_BGR2GRAY)
    # image_regularization(gray)
    image_enhance_gama(np_min)
    cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("max", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("gray", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("min", cv2.WINDOW_FREERATIO)

    cv2.imshow("image", imread)
    cv2.imshow("max", np_asarray1)
    cv2.imshow("min", np_asarray2)
    cv2.imshow("gray", my_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run():
    for file in files:
        read_image(file)

run()