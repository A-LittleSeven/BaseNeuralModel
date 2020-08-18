import tensorflow as tf
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D, Softmax, BatchNormalization, Conv2D, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, Adam
from keras.activations import relu, sigmoid
from keras.callbacks import TensorBoard
from tqdm import tqdm


class vgg16(object):
    def __init__(self):
        pass

    def model(self):
        model = Sequential()
        # input shape 224 * 224 * 3
        model.add(Conv2D(filters=64, input_shape=(224, 224, 3), kernel_size=(3, 3), strides=1, activation=relu,
                         padding="same", kernel_initializer="normal"))  # 224 * 224 * 3
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 224 * 224 * 3
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # 112 * 112 * 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 112 * 112 * 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 112 * 112 * 3
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # 56 * 56 * 3
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 56 * 56 * 3
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 56 * 56 * 3
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # 28 * 28 * 3
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 28 * 28 * 3
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 28 * 28 * 3
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 28 * 28 * 3
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # 14 * 14 * 3
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 28 * 28 * 3
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 28 * 28 * 3
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                         padding="same", activation=relu, kernel_initializer="normal"))  # 28 * 28 * 3
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # 7 * 7 * 3
        for i in range(2):
            model.add(Dense(4096, activation=relu))
        model.add(Dense(1000, activation=relu))
        model.add(Softmax())

        adam = Adam(lr=0.001)
        model.compile(
            optimizer=adam,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )
        model.summary()
        return model


if __name__ == '__main__':
    vgg = vgg16()
    vgg.model()
