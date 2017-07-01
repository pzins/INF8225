from __future__ import print_function
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K

import numpy as np
from numpy import random
from PIL import Image
import glob
import cv2


# use haarcascade to get the face
casc = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc)


# make the predictions from the model
def makePredictions(img, model_age, model_gender):
	x_train = np.array([np.array(img)])
	x_train = x_train.astype('float32')
	x_train /= 255
	pred_age = model_age.predict(x_train)
	pred_gender = model_gender.predict(x_train)
	print(np.argmax(pred_gender), pred_age)



model_age = load_model("model_age.h5")
model_gender = load_model("model_gender.h5")

# isolate the face from the img (webcam frame)
def getFace(img):
	faces = faceCascade.detectMultiScale(
			img,
			scaleFactor=1.1,
			minNeighbors=10,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in faces:
		im = img[y:y+h,x:x+w]
		im = cv2.resize(im, (128, 128))
		# print and waitKey to check it is working
		# cv2.imshow("Face found", im)
		# cv2.waitKey(0)
		return True, im
	return False, 0

# use the webcam
cam = cv2.VideoCapture(0)
counter = 0
while True:
	ret_val, img = cam.read()
	if cv2.waitKey(1) == 27:
		break
	img = cv2.flip(img, 1)
	if counter == 5:
		res, face = getFace(img)
		if res == True:
			makePredictions(face, model_age, model_gender)
		counter = 0
	cv2.imshow("webcam", img)
	counter += 1
