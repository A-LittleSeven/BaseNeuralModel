import tensorflow as tf
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D, Softmax, BatchNormalization, Conv2D, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, Adam
from keras.activations import relu, sigmoid, softmax
from keras.callbacks import TensorBoard
from tqdm import tqdm


class lenet5(object):
    def __init__(self):
        pass

    def model(self):
        model = Sequential()
        model.add(Conv2D(filters=6, input_shape=(32, 32, 1), kernel_size=(5, 5), padding="valid", strides=(1, 1),
                         activation=relu, kernel_initializer="uniform"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=16, kernel_size=(5, 5), padding="valid", strides=(1, 1),
                         activation=relu, kernel_initializer="uniform"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=120, kernel_size=(5, 5), padding="valid", strides=(1, 1),
                         activation=relu, kernel_initializer="uniform"))
        model.add(Dense(84, activation=relu))
        model.add(Dense(10, activation=softmax, trainable=False))
        
        adam = Adam(lr=0.001)
        model.compile(
            optimizer=adam,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )
        model.summary()
        return model

if __name__ == '__main__':
    lenet = lenet5()
    lenet.model()


