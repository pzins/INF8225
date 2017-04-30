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
from keras import optimizers
from keras.models import load_model


from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

#3 shifted class as in the paper Apparent Age Estimation Using Ensemble of Deep Learning Models
"""
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

num_classes = int(100/interval_length)+2
"""


num_classes = 7

# return age category
def getAgeCategory(age):
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


# load data
x_set = np.array([]).reshape(0, 128, 128, 3)
y_set = np.array([]).reshape(0)
for it in range(6):
    x_tmp = np.load("data/x_128_" + str(it) + ".dat")
    y_tmp = np.load("data/y_128_" + str(it) + ".dat")
    x_set = np.append(x_set, x_tmp, axis=0)
    y_set = np.append(y_set, y_tmp, axis=0)

# get age category from age
for i in range(len(y_set)):
  y_set[i] = getAgeCategory(y_set[i])

# transform to onehot vector
y_set = keras.utils.to_categorical(y_set, num_classes)


# split training, validation and test set
trainSize = int(x_set.shape[0] * 0.7)
validSize = int(x_set.shape[0] * 0.15)

x_train = x_set[:trainSize]
y_train = y_set[:trainSize]
x_val = x_set[trainSize:trainSize+validSize]
y_val = y_set[trainSize:trainSize+validSize]
x_test = x_set[trainSize+validSize:]
y_test = y_set[trainSize+validSize:]

epochs = 25
batch_size = 32
input_shape = (128, 128, 3)
data_augmentation = False



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255


# CNN network
activation = "relu"
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation,
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(100, activation=activation))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation="softmax"))


# optimizer
opt = optimizers.Adam(lr=0.001)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            shuffle=True)
  
  score = model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  # save model
  # model.save('age_classification.h5')
  
  
  pred = model.predict(x_test)
  for i in range(len(pred)):
  	print(str(pred[i]) + " => " + str(y_test[i]))
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False )  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                       batch_size=batch_size),
                         steps_per_epoch=x_train.shape[0] // batch_size,
                         epochs=epochs,
                         validation_data=(x_val, y_val))
  
  score = model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
    
  # save model
  # model.save('age_classification.h5')  
  
  pred = model.predict(x_test)
  for i in range(len(pred)):
    print(str(pred[i]) + " => " + str(y_test[i]))
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])