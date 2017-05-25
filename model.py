"""
function to solve Behavior cloning project for Udacity's nano-degree
"""

# import necessary modules
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import skimage.transform as sktransform
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
from preprocessing import *

# parameters
LEARN_RATE = 2e-4
BATCH_SIZE = 128
EPOCHS = 15

###################################
## define model architechture
# basic neural network to predict steering str_angle
def model_architechture(img_shape=(66, 200, 3)):
    # number_of_epochs = 8
    activation_relu = 'relu'

    # initialize
    model = Sequential()
    # Normalize
    print(img_shape)
    # Normalize
    model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=img_shape))
    # model.add(Cropping2D(cropping=((70, 20),(0, 0))))
    # model.add(Cropping2D(cropping=((0, 0), (80, 20))))

    # valid border mode should get rid of a couple each way, whereas same keeps
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())
    model.add(Dropout(.5))

    model.add(Dense(1164, W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(Dropout(.5))

    model.add(Dense(500, W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))

    # model.add(Dense(200, activation='relu'))
    # # model.add(Dropout(.7))

    # model.add(Dense(100))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(Dropout(.5))

    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    model.add(Dropout(.25))

    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Activation(activation_relu))
    # model.add(Dropout(.25))

    model.add(Dense(1))

    model.summary()

    return model

####################################
# generator for validation data
def valid_generator(X, y, batch_size, num_per_epoch):
    while True:
        # X, y = shuffle(X, y)
        smaller = min(len(X), num_per_epoch)
        iterations = int(smaller / batch_size)
        for i in range(iterations):
            start, end = i * batch_size, (i + 1) * batch_size

            X_new, y_new = process_data_set(X[start: end], y[start: end])

            yield(X_new, y_new)

###################################
# generator function
def train_generator(X, y, batch_size, num_per_epoch):
    while True:
        X, y = shuffle(X, y)
        # print('range is', int(num_per_epoch/batch_size))
        smaller = min(len(X), num_per_epoch)
        iterations = int(smaller / batch_size)
        for i in range(iterations):
            start, end = i * batch_size, (i + 1) * batch_size

            X_new, y_new = process_data_set(X[start: end], y[start: end])

            yield(X_new, y_new)


###################################
############## main script
if __name__ == "__main__":
    # define training set
    images, str_angles = read_and_add_training_data()

    X_train, X_val, y_train, y_val = \
        train_test_split(images, str_angles, test_size=.2, random_state=0)

    print('X_train and y_train', X_train.shape, y_train.shape)
    print('X_val shape', X_val.shape)

    # # compile and train the model using the generator function
    # train_generator = generator(X_train, y_train, batch_size=BATCH_SIZE)
    # validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    # define model-architechture
    model = model_architechture()

    model.compile(optimizer=Adam(LEARN_RATE), loss="mse", )

    model.fit_generator(train_generator(X_train, y_train, batch_size=BATCH_SIZE, num_per_epoch=len(y_train)),
                        samples_per_epoch=len(X_train),
                        validation_data=valid_generator(X_val, y_val, batch_size=BATCH_SIZE, num_per_epoch=len(y_val)),
                        nb_val_samples=len(y_val), nb_epoch=EPOCHS)

    # print the keys contained in the history object
    # print(history_object.history.keys())

    print("DONE")

    model.save('./model.h5')






