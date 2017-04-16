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

#CNN
#------------------------------------------------------

"""
index = np.arange(x.shape[0])
np.random.shuffle(index)

sizeTrain = int(x.shape[0] * 0.6)
sizeValTest = int(x.shape[0] * 0.2)

x_train = x[index[:sizeTrain]]
x_valid = x[index[sizeTrain:sizeTrain+sizeValTest]]
x_test = x[index[sizeTrain+sizeValTest:sizeTrain+sizeValTest*2]]
y_train = y[index[:sizeTrain]]
y_valid = y[index[sizeTrain:sizeTrain+sizeValTest]]
y_test = y[index[sizeTrain+sizeValTest:sizeTrain+sizeValTest*2]]
"""
# group age into categories
# -1 => error
# 0 => <20
# 1 => 20-30
# 2 => 30-40
# 3 => 40-50
# 4 => 50-60
# 5 => >60

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
for it in range(6):
    x_tmp = np.load("data1000/128_age_gender/xtrain_128_" + str(it) + ".dat")
    y_tmp = np.load("data1000/128_age_gender/ytrain_128_" + str(it) + ".dat")
    x_set = np.append(x_set, x_tmp, axis=0)
    y_set_age = np.append(y_set_age, y_tmp, axis=0)
    y_set_gender = np.append(y_set_gender, y_tmp, axis=0)

y_set_age = np.delete(y_set_age, 0, 1)
y_set_gender = np.delete(y_set_gender, -1, 1)

for i in range(len(y_set_age)):
  y_set_age[i] = getAgeCategory(y_set_age[i])

# y_set_age = keras.utils.to_categorical(y_set_age, num_classes_age)
y_set_gender = keras.utils.to_categorical(y_set_gender, num_classes_gender)


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

epochs_gender = 3
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

# model = load_model("keras_model.h5")
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
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
"""
"""
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
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
"""


age_model = Sequential()
age_model.add(model)
# age_model.add(Dense(num_classes_age, activation="softmax"))
age_model.add(Dense(1))

gender_model = Sequential()
gender_model.add(model)
gender_model.add(Dense(num_classes_gender, activation="sigmoid"))


opt = optimizers.Adam(lr=0.001)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

"""
age_model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])
"""
age_model.compile(loss="mean_squared_error",
              optimizer=opt,
              metrics=['mae'])
gender_model.compile(loss="categorical_crossentropy",#keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:
  print("Train gender model")
  gender_model.fit(x_train, y_train_gender,
            batch_size=batch_size,
            epochs=epochs_gender,
            verbose=1,
            validation_data=(x_test, y_test_gender),
            shuffle=True)
  print("Train age model")
  age_model.fit(x_train, y_train_age,
            batch_size=batch_size,
            epochs=epochs_age,
            verbose=1,
            validation_data=(x_test, y_test_age),
            shuffle=True)
  
  score = gender_model.evaluate(x_val, y_val_gender, verbose=1)
  print('(Gender) Test loss:', score[0])
  print('(Gender) Test accuracy:', score[1])

  score = age_model.evaluate(x_val, y_val_age, verbose=1)
  print('(Age) Test loss:', score[0])
  print('(Age) Test accuracy:', score[1])
  
  # model.save('keras_model_age.h5')
  
  
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