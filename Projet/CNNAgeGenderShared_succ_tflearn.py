from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from numpy import random
from PIL import Image
import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.objectives import *
import tensorflow as tf

num_classes_gender = 2

def getAgeCategory(age):  
  return age

x_set = np.array([]).reshape(0, 128, 128, 3)
y_set_age = np.array([]).reshape(0,2)
y_set_gender = np.array([]).reshape(0,2)
for it in range(1):
    x_tmp = np.load("data/x_128_" + str(it) + ".dat")
    y_tmp = np.load("data/y_128_" + str(it) + ".dat")
    x_set = np.append(x_set, x_tmp, axis=0)
    y_set_age = np.append(y_set_age, y_tmp, axis=0)
    y_set_gender = np.append(y_set_gender, y_tmp, axis=0)

y_set_age = np.delete(y_set_age, 0, 1)
y_set_gender = np.delete(y_set_gender, -1, 1)

for i in range(len(y_set_age)):
  y_set_age[i] = getAgeCategory(y_set_age[i])

y_set_gender = keras.utils.to_categorical(y_set_gender, num_classes_gender)

y_set = np.column_stack((y_set_age, y_set_gender))


trainSize = int(x_set.shape[0] * 0.7)
validSize = int(x_set.shape[0] * 0.15)
x_train = x_set[:trainSize]
y_train_age = y_set_age[:trainSize]
y_train_gender = y_set_gender[:trainSize]

x_val = x_set[trainSize:trainSize+validSize]
y_val_gender = y_set_gender[trainSize:trainSize+validSize]
y_val_age = y_set_age[trainSize:trainSize+validSize]

x_test = x_set[trainSize+validSize:]
y_test_age = y_set_age[trainSize+validSize:]
y_test_gender = y_set_gender[trainSize+validSize:]

batch_size = 64
input_shape = (128, 128, 3)
data_augmentation = False
epochs = 1


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=5)
img_aug.add_random_blur (sigma_max=5.0)

# Convolutional network building
network = input_data(shape=input_shape)
                     # data_preprocessing=img_prep,
                     # data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network_age = fully_connected(network, 128, activation='relu')
network_age = dropout(network_age, 0.5)
network_age = fully_connected(network_age, 1)
network_age = regression(network_age, optimizer='adam',
                     loss="mean_square",
                     learning_rate=0.001)

network_gender = fully_connected(network, 128, activation='relu')
network_gender = dropout(network_gender, 0.5)
network_gender = fully_connected(network_gender, 2)
network_gender = regression(network_gender, optimizer='adam',
                     loss=softmax_categorical_crossentropy,
                     learning_rate=0.001)

model_gender = tflearn.DNN(network_gender, tensorboard_verbose=3)
model_age = tflearn.DNN(network_age, tensorboard_verbose=3)

# model.load("mymodel.tflearn")
model_gender.fit(x_train, y_train_gender, n_epoch=20, shuffle=True, validation_set=(x_val, y_val_gender),
          show_metric=True, batch_size=32, run_id='gender')  

model_age.fit(x_train, y_train_age, n_epoch=20, shuffle=True, validation_set=(x_val, y_val_age),
          show_metric=True, batch_size=32, run_id='age')  

# model.save("mymodel.tflearn")

x_test = x_test[:100]
x_test = x_test[:100]

e = model_age.evaluate(x_test, y_test_age)
print(e)
p = model_age.predict(x_test)
for i in range(len(p)):
  print(str(p[i]) + " => " + str(y_test_age[i]))

