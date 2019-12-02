import tensorflow as tf
import cv2
import numpy as np
import model
import socket
import threading
import sys
import os
import struct

x, y, predict, loss, accuracy, merged, softmax = model.build_model()
saver = tf.train.Saver()
sess = tf.InteractiveSession()

saver.restore(sess, tf.train.latest_checkpoint("./persist_model_1/"))


def predict_value(type='image', image_path=None):

    if type == 'image':
        assert image_path is not None
        image = cv2.imread(image_path)
        if image.shape != (200, 200):
            image = cv2.resize(image, (200, 200))
        image = np.asarray(image, np.float32) / 255.
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        [predict_, probabilistic] = sess.run([predict, softmax], feed_dict={x: image})
        probabilistic = np.max(probabilistic)
        print("category: %d, prob: %.4f" % (predict, probabilistic))
        if predict_ == 0:
           return 2 #有毒垃圾
        elif predict_ == 1:
            return 1 #不可回收垃圾
        elif predict_ == 2:
            return 3 #厨余垃圾
        elif predict_ == 3:
            return 0 #可回收
        elif predict_ == 4:
            return 4 #空盒子
        elif predict_ == 5:
            return 0 # 可回收垃圾
        elif predict_ == 6:
            return 1 # 不可回收垃圾

    elif type == 'video':
        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            resize = cv2.resize(frame, (200, 200))
            x_ = np.asarray(resize, np.float32) / 255.
            x_ = np.reshape(x_, [1, x_.shape[0], x_.shape[1], x_.shape[2]])
            [predict_, probabilistic] = sess.run([predict, softmax], feed_dict={x: x_})
            probabilistic = np.max(probabilistic)
            if predict_ == 0:
                cv2.putText(frame, "electric probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 2, cv2.LINE_AA)
            elif predict_ == 1:
                cv2.putText(frame, "fruit probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (99, 255, 255), 2, cv2.LINE_AA)
            elif predict_ == 2:
                cv2.putText(frame, "lajiao probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 12, 130), 2, cv2.LINE_AA)
            elif predict_ == 3:
                cv2.putText(frame, "yilaguan probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (19, 15, 255), 2, cv2.LINE_AA)
            elif predict_ == 4:
                cv2.putText(frame, "void hezi probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 190, 255), 2, cv2.LINE_AA)
            elif predict_ == 5:
                cv2.putText(frame, "zhi probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 70, 13), 2, cv2.LINE_AA)
            elif predict_ == 6:
                cv2.putText(frame, "jiaobu probabilistic: %.4f" % probabilistic, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 100), 2, cv2.LINE_AA)

            cv2.imshow("recognized", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
        capture.release()
        __close_session__(sess)


def __close_session__(sess):
    sess.close()


def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 6666))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sess.close()
        sys.exit(1)
    print('Waiting connection...')

    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()


def deal_data(conn, addr):
    global predict
    print('Accept new connection from {0}'.format(addr))
    # conn.settimeout(500)
    conn.send('Hi, Welcome to the server!'.encode("utf-8"))

    while 1:
        fileinfo_size = struct.calcsize('128sl')
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.strip(b"\x00").decode("utf-8")
            new_filename = os.path.join("./picturetest/test.jpg")
            print(new_filename, filesize)
            print('file new name is {0}, filesize if {1}'.format(new_filename, filesize))

            recvd_size = 0  # 定义已接收文件的大小
            fp = open(new_filename, 'wb')
            print('start receiving...')

            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print('end receive...')
        conn.send('已上传'.encode("utf-8"))
        try:
            print("进入识别函数")
            value = predict_value("image", "./picturetest/test.jpg")
            print("函数执行完成：", value)
            conn.send('识别结果：{0}'.format(value).encode("utf-8"))
        except Exception as e:
            print(e)
            sess.close()
            conn.send('识别失败'.encode("utf-8"))
        conn.close()
        break


if __name__ == '__main__':
    # predict_value("video")
    socket_service()