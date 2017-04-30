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
y_val_age = y_set_age[trainSize:trainSize+validSize]
y_val_gender = y_set_gender[trainSize:trainSize+validSize]

x_test = x_set[trainSize+validSize:]
y_test_age = y_set_age[trainSize+validSize:]
y_test_gender = y_set_gender[trainSize+validSize:]

epochs_gender = 10
epochs_age = 20
batch_size = 32
input_shape = (128, 128, 3)
data_augmentation = False


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


age_model = Sequential()
age_model.add(model)
age_model.add(Flatten())
age_model.add(Dense(128, activation='relu'))
age_model.add(Dropout(0.5))
age_model.add(Dense(1))

gender_model = Sequential()
gender_model.add(model)
gender_model.add(Flatten())
gender_model.add(Dense(128, activation='relu'))
gender_model.add(Dropout(0.5))
gender_model.add(Dense(num_classes_gender, activation="sigmoid"))


opt = optimizers.Adam(lr=0.001)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

age_model.compile(loss="mean_squared_error",
              optimizer=opt,
              metrics=['mae'])
gender_model.compile(loss="categorical_crossentropy",#keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:
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


  print("Train gender model")
  gender_model.fit_generator(datagen.flow(x_train, y_train_gender,
                     batch_size=batch_size),
                       steps_per_epoch=x_train.shape[0] // batch_size,
                       epochs=epochs_gender,
                       validation_data=(x_val, y_val_gender))

  print("Train age model")
  age_model.fit_generator(datagen.flow(x_train, y_train_age,
                   batch_size=batch_size),
                     steps_per_epoch=x_train.shape[0] // batch_size,
                     epochs=epochs_age,
                     validation_data=(x_val, y_val_age))
  score = age_model.evaluate(x_val, y_val_age, verbose=1)
  print('(Age) Test loss:', score[0])
  print('(Age) Test accuracy:', score[1])
  exit()
  
  
  pred = gender_model.predict(x_test)
  # print(pred)
  for i in range(len(pred)):
  	print(str(pred[i]) + " => " + str(y_test_gender[i]))
  print('(Gender) Test loss:', score[0])
  print('(Gender) Test accuracy:', score[1])
  
  pred = age_model.predict(x_test)
  # print(pred)
  for i in range(len(pred)):
    print(str(pred[i]) + " => " + str(y_test_age[i]))
  print('(Age) Test loss:', score[0])
  print('(Age) Test accuracy:', score[1])


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
    pred = model.predict(x_test)
    print(pred)
  