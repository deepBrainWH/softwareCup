import matplotlib.pyplot as plt
import numpy as np
import cv2


def grayHist(image):
    h, w = image.shape[:2]
    pixelSequence = image.reshape([h * w, -1])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()


def image_enhance_1(in_image):
    grayHist(in_image)
    out_image = in_image * 2
    out_image[out_image > 255] = 255
    grayHist(out_image)
    cv2.imshow("out_image", out_image)


def image_regularization(in_image):
    imin, imax = np.min(in_image), np.max(in_image)
    omin = 0
    omax = 255
    a = (omax - omin) / (imax - imin)
    b = omin - a * imin
    out = np.asarray(a * in_image + b, dtype=np.uint8)
    grayHist(in_image)
    grayHist(out)
    cv2.imshow("out", out)


def image_enhance_gama(in_image):
    """
    假设输入图像为I，宽为W，高为H，首先将其灰度值归一化到[0,1]范围，
    对于8位图来说，除以255即可。I(r,c)代表归一化后的第r行第c列的灰度
    值，输出图像记为O，伽马变换就是O(r,c) = I(r,c)^gamma
    图像不变。如果图像整体或者感兴趣区域较暗，则令0< gamma <1可以增加图像对比度；
    相反，如果图像整体或者感兴趣区域较亮，则令gamma >1可以降低图像对比度。
    :param in_image:
    :return:
    """
    in_image = in_image / 255.
    gamma = 0.3
    out = np.power(in_image, gamma)
    out = np.asarray(out*255, dtype=np.uint8)
    print(np.max(out), np.min(out))
    cv2.imshow("gamma", out)
    grayHist(out)
