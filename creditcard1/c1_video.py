import cv2
import numpy as np

def video():
    capture = cv2.VideoCapture(0)
    masker = np.zeros((480, 640, 3), 'uint8')
    cv2.rectangle(masker, (120, 110), (520, 370), (255, 255, 255), -1)
    while True:
        ret, frame = capture.read()
        cv2.namedWindow("frame", cv2.WINDOW_FREERATIO)
        frame = cv2.bitwise_and(frame, masker)
        cv2.putText(frame, "Please put your bank card in the designated area!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(frame, (150, 260), (490, 290), (255, 0, 0), 1, cv2.LINE_8)
        cv2.imshow("frame", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
        elif c == 97:
            image = frame[260:290, 150:490]
            cv2.imshow("image", image)
    capture.release()
    cv2.destroyAllWindows()

def predict(image):
    rows, cols= image.shape[0], image.shape[1]
    w = 41 / rows


if __name__ == '__main__':
    video()