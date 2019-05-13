import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from tensorflow.keras.callbacks import TensorBoard
import time




X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
# Normalize data to 0-1. Use keras.utils.normalize otherwise
X = X / 255.0

# Setting HyperParameters 
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

            model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])

            model.fit(X, y, epochs = 10, batch_size=64, validation_split=0.1, callbacks=[tensorboard])


