from keras import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU


class YoloV1(object):
    num_of_class = 4

    classes = [
        'person',
        'smartphone',
        'bottle',
        'car'
    ]

    def __init__(self, leaky_relu=0.1):
        self.leaky_relu = leaky_relu

    def build_model(self):
        model = Sequential()
        # layer1
        model.add(Convolution2D(16, (3, 3), input_shape=(448, 448, 3), padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(MaxPooling2D((2, 2)))

        # Layer 2
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        # Layer 3
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        # Layer 4
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        # Layer 5
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        # Layer 6
        model.add(Convolution2D(512, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        # Layer 7
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        # Layer 8
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        # Layer 9
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())

        # Layer 10
        model.add(Dense(256))

        # Layer 11
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))

        # Layer 12
        model.add(Dense(1470))

        return model
