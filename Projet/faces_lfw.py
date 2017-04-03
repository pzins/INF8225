from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from PIL import Image
import glob
#CNN
#------------------------------------------------------

import cv2
import numpy as np
import glob
from PIL import Image
import scipy.io as sio
import datetime
from datetime import *


label = open('lfw/gen.txt','r')
labels = []
c = 0
for i in label.read():
	if i == '0':
		labels.append(0)
	else:
		labels.append(1)
	c += 1
y_train = np.array(labels)


taille = 100

fl = glob.glob("/home/pierre/Dev/CNN/lfw/all/*.jpg")
x_train = np.array([np.array(Image.open(fname)) for fname in fl[:taille+1]])



casc = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc)

real_set = []
label = []
counter = 0
for img in x_train:
	# img = cv2.imread(image)
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = faceCascade.detectMultiScale(
		img,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in faces:
		# print(w,h)
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2)
		im = img[y:y+h,x:x+w]
		im = cv2.resize(im, (50, 50))
		real_set.append(im)

		break

	counter += 1
	# cv2.imshow("Face found", im)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

x_train = np.array(real_set)
y_train = keras.utils.to_categorical(y_train, 2)



#CNN
#-------------------------------------------------------------------------------------------


batch_size = 128
num_classes = 10
epochs = 12
input_shape = (50,50,3)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
num_classes = 2
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
          # validation_data=(x_test, y_test))
"""
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
pred = model.predict(x_test)
for i in pred:
	print(i)
"""
print("*******")