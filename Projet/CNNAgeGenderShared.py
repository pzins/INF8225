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

from keras.losses import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Activation, Dense, Merge
from keras import backend as K
from keras.metrics import *

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

# model = load_model("keras_model.h5")
_model = Sequential()
_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 kernel_initializer="he_normal"))
_model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer="he_normal"))
_model.add(MaxPooling2D(pool_size=(2, 2)))
_model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer="he_normal"))
_model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer="he_normal"))
_model.add(MaxPooling2D(pool_size=(2, 2)))
_model.add(Dropout(0.25))
_model.add(Flatten())
_model.add(Dense(128, activation='relu', kernel_initializer="he_normal"))
_model.add(Dropout(0.5))
_model.add(Dense(3))

a = Sequential()
a.add(_model)
a.add(Dense(1))

b = Sequential()
b.add(_model)
b.add(Dense(2, activation="softmax"))


model = Sequential()
model.add(Merge([a, b], mode = 'concat'))
# model.add(Dense(3))
print(y_val.shape)
# exit()

def _loss_tensor(y_true, y_pred):
  return mean_squared_error(y_true[0], y_pred[0]) + categorical_crossentropy(y_true[1:3], y_pred[1:3])

    # y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    # out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    # return K.mean(out, axis=-1)

def mean_pred(y_true, y_pred):
  return categorical_accuracy(y_true[1:3], y_pred[1:3])
  return mean_absolute_error(y_true[0], y_pred[0])
  return mean_absolute_error(y_true[0], y_pred[0])


opt = optimizers.Adam(lr=0.001)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=_loss_tensor,
              optimizer=opt,
              metrics=[mean_pred, "mae", "accuracy"])

if not data_augmentation:
  print("Train model")
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_val, y_val),
            shuffle=True)
  
  score = model.evaluate(x_val, y_val, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  # model.save('keras_model_age.h5')
  
  pred = model.predict(x_test)
  # print(pred)
  a = []
  b = []
  for i in range(len(pred)):
    print(str(pred[i]) + " => " + str(y_test[i]))
    a.append(pred[i][0])
    b.append(y_test[i][0])

  somm = 0
  for i in range(len(a)):
    somm += abs(float(a[i])-float(b[i]))
  print(somm/len(a))
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  print('Test mae:', score[2])


else:
  """
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
  """