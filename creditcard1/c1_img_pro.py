import cv2
import myconfig
import os

GAUSSIAN_BLUR_KERNEL_SIZE = 9

directory = myconfig.credit_card_dir
listdir = os.listdir(directory)
for image in listdir:
    img_path = os.path.join(directory, image)
    if os.path.isfile(img_path):
        image = cv2.imread(img_path)
        image = cv2.GaussianBlur(image, (GAUSSIAN_BLUR_KERNEL_SIZE,GAUSSIAN_BLUR_KERNEL_SIZE), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 40, 70)
        cv2.findContours(canny, )
        cv2.imshow("canny", canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        continue
