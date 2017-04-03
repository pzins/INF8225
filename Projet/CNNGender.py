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

from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#CNN
#------------------------------------------------------

x = np.load("xtrain_50.dat")
y = np.load("ytrain_50.dat")

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

epochs = 5
batch_size = 100
num_classes = 2
input_shape = (50,50,1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='tanh',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='tanh'))
model.add(Conv2D(64, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='tanh'))
# model.add(Conv2D(64, (3, 3), activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = optimizers.Adam(lr=0.01)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred = model.predict(x_test)
for i in range(len(pred)):
	print(str(pred[i]) + " => " + str(y_test[i]))
