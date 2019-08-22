import tensorflow as tf
import cv2
import numpy as np
import model

sess = None


def __init_session():
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, tf.train.latest_checkpoint("./h5_dell/"))


def predict_value(type='image', image_path=None):
    x, y, predict, loss, accuracy, merged, softmax = model.build_model()
    if type == 'image':
        assert image_path is not None
        image = cv2.imread(image_path)
        if image.shape != (200, 200):
            image = cv2.resize(image, (200, 200))
        image = np.asarray(image, np.float32) / 255.
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        [predict, probabilistic] = sess.run([predict, softmax], feed_dict={x: image})
        probabilistic = np.max(probabilistic)
        print("category: %d, prob: %.4f" % (predict, probabilistic))

    elif type == 'video':
        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            resize = cv2.resize(frame, (200, 200))
            x_ = np.asarray(resize, np.float32) / 255.
            x_ = np.reshape(x_, [1, x_.shape[0], x_.shape[1], x_.shape[2]])
            [predict, probabilistic] = sess.run([predict, softmax], feed_dict={x: x_})
            probabilistic = np.max(probabilistic)
            if predict == 0:
                cv2.putText(frame, "0 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
            elif predict == 1:
                cv2.putText(frame, "1 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2, cv2.LINE_AA)
            elif predict == 2:
                cv2.putText(frame, "2 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)
            elif predict == 3:
                cv2.putText(frame, "3 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 2, cv2.LINE_AA)
            elif predict == 4:
                cv2.putText(frame, "4 probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("recognized", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
        capture.release()


def __close_session():
    sess.close()


if __name__ == '__main__':
    __init_session()
    predict_value("video")
    __close_session()
