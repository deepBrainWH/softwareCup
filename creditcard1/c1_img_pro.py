import cv2
import myconfig
import os


directory = myconfig.credit_card_dir
listdir = os.listdir(directory)
for image in listdir:
    img_path = os.path.join(directory, image)
    if os.path.isfile(img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (7,7), 0)
        ret, ostu = cv2.threshold(image, 150, 255, cv2.THRESH_OTSU)
        threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,27, 2)
        cv2.imshow("means", threshold)
        cv2.imshow("ostu", ostu)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        continue
