from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import multi_gpu_model
import c1_dataset

IMAGE_ROWS = 41
IMAGE_COLS = 28
IMAGE_DIM = 3
NUM_CLASS = 11


def model(multi_gpu=False, gpus=4):
    # load all data
    mybatch = c1_dataset.batch()
    x_train, y_train, x_test, y_test = mybatch.load_data()
    print("x_train_shape:", x_train.shape, '\n', 'y_train_shape:', y_train.shape)
    print('x_test_shape:', x_test.shape, '\n', 'y_test_shape:', y_test.shape)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_DIM)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASS, activation='softmax'))
    if multi_gpu:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=Adadelta(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
<<<<<<< HEAD
    filepath = "./h5/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
=======
    filepath = "./h5_deep_server/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5_deep_server"
>>>>>>> d80a7d0f8e07fdf0e0c8a5bd8bad549119f7a956
    callback_list = [TensorBoard('./log'), ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)]
    model.fit(x_train, y_train, 128, 1000, 1, callbacks=callback_list, validation_split=0.3)


def load_keras_model(model_path):
    return load_model(model_path)

if __name__ == '__main__':
    model()
<<<<<<< HEAD
    # mymodel = load_keras_model("./h5/weights-improvement-116-0.98.h5")
=======
    # mymodel = load_keras_model("./h5_deep_server/weights-improvement-116-0.98.h5_deep_server")
>>>>>>> d80a7d0f8e07fdf0e0c8a5bd8bad549119f7a956
    # while True:
    #     filename = input("please input one file name:")
    #     filepath = "C:\\Users\\wangheng\\Documents\\software_cup\\number\\" + filename
    #     print(filepath)
    #     if os.path.isfile(filepath):
    #         imread = cv2.imread(filepath)
    #         reshape = np.reshape(imread, (1, imread.shape[0], imread.shape[1], imread.shape[2]))
    #         data_ = np.asarray(reshape,np.float32) / 255.0
    #         result = mymodel.predict(data_)
    #         res = np.argmax(result)
    #         if res == 10:
    #             print("not a number!")
    #         else:
    #             print(res)
    #     else:
    #         print("file path error!")