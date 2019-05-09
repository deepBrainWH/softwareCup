import cv2
import myconfig
import os

GAUSSIAN_BLUR_KERNEL_SIZE = 9

directory = myconfig.credit_card_dir
listdir = os.listdir(directory)


def process1():
    for image in listdir:
        img_path = os.path.join(directory, image)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            image = cv2.GaussianBlur(image, (GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), 0)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thre = cv2.threshold(gray, 25, 255, cv2.THRESH_OTSU)
            # canny = cv2.Canny(gray, 40, 70)
            # cv2.findContours(canny, )
            cv2.imshow("canny", thre)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            continue


def process2():
    number_dir = myconfig.number_bar_dir
    list_number_dir = os.listdir(number_dir)

    for img_name in list_number_dir:
        img_path = number_dir + img_name
        image = cv2.imread(img_path)
        image = cv2.GaussianBlur(image, (7, 7), 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 0)
        canny = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 80)
        dst = cv2.resize(image, (int(41. / image.shape[0] * image.shape[1]), 41))
        numbers = dst.shape[1] // 24
        start = 3
        end = dst.shape[1]
        while start < end:
            cv2.rectangle(dst, (start, 2), (start + 21, 38), (0,0,255))
            start += 25
        cv2.imshow("canny", gray)
        cv2.imshow("adap", adap)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    process2()
