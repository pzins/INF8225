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



age_interval = 101
interval_length = 10
ages = np.arange(age_interval)
classes1 = np.zeros(age_interval)
classes2 = np.zeros(age_interval)
classes3 = np.zeros(age_interval)
cur_class = 1
counter = 0

for i in range(1, len(ages)):
  classes1[i] = cur_class
  counter += 1
  if counter == interval_length:
    cur_class += 1
    counter = 0

cur_class = 1
for i in range(2, len(ages)):
  classes2[i] = cur_class
  counter += 1
  if counter == interval_length:
    cur_class += 1
    counter = 0

cur_class = 1
for i in range(3, len(ages)):
  classes3[i] = cur_class
  counter += 1
  if counter == interval_length:
    cur_class += 1
    counter = 0
# les trois classes shifted comme ds l'article



# num_classes = int(100/interval_length)+2
num_classes_age = 7
num_classes_gender = 2

def getAgeCategory(age):
  """
  if age<50:
    return 0
  return 1
  """
  # return classes1[int(age)]
  
  return age

  if age < 20:
    return 0
  elif age >= 20 and age < 25:
    return 1
  elif age >= 25 and age < 30:
    return 2
  elif age >= 30 and age < 35:
    return 3
  elif age >= 35 and age < 45:
    return 4
  elif age >= 45 and age < 60:
    return 5
  else:
    return 6


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

# y_set_age = keras.utils.to_categorical(y_set_age, num_classes_age)
y_set_gender = keras.utils.to_categorical(y_set_gender, num_classes_gender)

y_set = np.column_stack((y_set_age, y_set_gender))


trainSize = int(x_set.shape[0] * 0.7)
validSize = int(x_set.shape[0] * 0.15)

x_train = x_set[:trainSize]
y_train = y_set[:trainSize]

x_val = x_set[trainSize:trainSize+validSize]
y_val = y_set[trainSize:trainSize+validSize]

x_test = x_set[trainSize+validSize:]
y_test = y_set[trainSize+validSize:]

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

def myLoss(prediction, target):
    a =  mean_square(prediction[0], target[0])
    print(a.shape)
    return a
    return softmax_categorical_crossentropy(prediction[1:3], target[1:3])
    return 0.8*mean_square(prediction[0], target[0]) + 0.2*softmax_categorical_crossentropy(prediction[1:3], target[1:3])

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
network = dropout(network, 0.75)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 3)
network = regression(network, optimizer='adam',
                     loss=myLoss,
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=3)

# model.load("mymodel.tflearn")

model.fit(x_train, y_train, n_epoch=10, shuffle=True, validation_set=(x_val, y_val),
          show_metric=True, batch_size=32, run_id='age_reg')  

# model.save("mymodel.tflearn")

x_test = x_test[:100]
x_test = x_test[:100]

e = model.evaluate(x_test, y_test)
print(e)
p = model.predict(x_test)
for i in range(len(p)):
  print(str(p[i]) + " => " + str(y_test[i]))

