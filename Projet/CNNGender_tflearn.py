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

#CNN
#------------------------------------------------------

x_set = np.array([]).reshape(0, 32, 32, 3) #(0, 50, 50, 1)
y_set = np.array([]).reshape(0, 2)
for it in range(6):
    x_tmp = np.load("data1000/32_large/xtrain_32_" + str(it) + ".dat")
    y_tmp = np.load("data1000/32_large/ytrain_32_" + str(it) + ".dat")
    x_set = np.append(x_set, x_tmp, axis=0)
    y_set = np.append(y_set, y_tmp, axis=0)

# x_set = np.squeeze(x_set)
print(x_set.shape)
print(y_set.shape)

trainSize = int(x_set.shape[0] * 0.7)
validSize = int(x_set.shape[0] * 0.25)

x_train = x_set[:trainSize]
y_train = y_set[:trainSize]
x_val = x_set[trainSize:trainSize+validSize]
y_val = y_set[trainSize:trainSize+validSize]
x_test = x_set[trainSize+validSize:]
y_test = y_set[trainSize+validSize:]

epochs = 100
batch_size = 32
num_classes = 2
input_shape = (32, 32, 3) #(50, 50, 1)
data_augmentation = True




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
# img_aug.add_random_blur (sigma_max=5.0)



# Convolutional network building
network = input_data(shape=input_shape,
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation="sigmoid")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=3)

# model.load("mymodel.tflearn")

model.fit(x_train, y_train, n_epoch=50, shuffle=True, validation_set=(x_val, y_val),
          show_metric=True, batch_size=512, run_id='gender_margin')  

model.save("CNNGender_32_large.tflearn")


e = model.evaluate(x_test, y_test)
print(e)

p = model.predict(x_test)
for i in range(len(p)):
  print(str(p[i]) + " => " + str(y_test[i]))

