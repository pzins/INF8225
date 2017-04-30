from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import scipy.io as sio
import scipy.misc

import numpy as np
from PIL import Image
import glob
import cv2
import datetime
from datetime import *


# get age
def getAge(dob, photo_date):
	dob = np.asscalar(dob)
	python_datetime = datetime.fromordinal(int(dob)) + timedelta(days=dob%1) - timedelta(days = 366)
	res = photo_date - python_datetime.year
	if res < 10 or res > 100:
		res = -1
	return res

# group age into categories
# -1 => error
# 0 => <20
# 1 => 20-30
# 2 => 30-40
# 3 => 40-50
# 4 => 50-60
# 5 => >60
# get age category (no more used, now categories are made later in CNNAge.py script)
def getAgeCategory(age):
	if age < 10 or age > 100:
		return -1
	elif age<20:
		return 0
	elif age >= 20 and age < 30:
		return 1
	elif age >= 30 and age < 40:
		return 2
	elif age >= 40 and age < 50:
		return 3
	elif age >= 50 and age < 60:
		return 4
	else:
		return 5


data = sio.loadmat("wiki.mat")
d = data["wiki"]
taille_img_out = 32
nb_categories = 2

a = np.array(d[0][0][2][:][0])

dataset_size = a.shape[0]
a = a[:60000]

batchSize = 10

for it in range(int(dataset_size/batchSize)):

	x_train = np.array([np.array(cv2.imread("/home/pierre/Downloads/wiki_crop/"+fname[0], 1)) for fname in a[it*batchSize:(it+1)*batchSize]])

	b = np.array(d[0][0][3][:][0][it*batchSize:(it+1)*batchSize])
	
	dob = np.array(d[0][0][0][:][0][it*batchSize:(it+1)*batchSize])
	photo_date = np.array(d[0][0][1][:][0][it*batchSize:(it+1)*batchSize])

	for i in range(len(dob)):
		# dob[i] = getAgeCategory(getAge(dob[i], photo_date[i]))
		dob[i] = getAge(dob[i], photo_date[i])
	
	# if we want both informations : gender and age
	b = np.column_stack((b,dob))


	casc = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(casc)

	real_set = []
	label = []
	counter = 0
	for img in x_train:
		
		if len(img.shape) == 2:
			counter += 1
			continue

		faces = faceCascade.detectMultiScale(
			img,
			scaleFactor=1.1,
			minNeighbors=10,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)
		test = 0
		for (x,y,w,h) in faces:
			
			# cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
			
			#enlarge bounding box
			y -= int(h*0.25)
			x -= int(w*0.25)
			h += int(h*0.25)
			w += int(w*0.25)

			im = img[y:y+h,x:x+w]
			
			if im.shape[0] < taille_img_out or im.shape[1] < taille_img_out:
				break

			im = cv2.resize(im, (taille_img_out, taille_img_out))
			real_set.append(im)
			test = 1

			# show image with face detection
			# cv2.imshow("Face found", img)
			# cv2.waitKey(0)

			# save img file	to see result
			# if counter < 250:	
				# name = "test/" + str(counter) + "_" + str(int(b[counter])) + '.jpg'
				# scipy.misc.imsave(name, im)
			break

		if test == 1:
			if np.isnan(b[counter][0]) or b[counter][1] == -1:
				real_set.pop()
			else:
				label.append(b[counter])

		counter += 1


	x_train = np.asarray(real_set)
	y_train = np.array(label)

	# for gender, we do one-hot vector
	# y_train = keras.utils.to_categorical(y_train, nb_categories)
	

	# x_train = np.expand_dims(x_train, 4)

	# save matrices : input data x and labels y
	x_train.dump("data/x_" + str(taille_img_out) + "_" + str(it) + ".dat")
	y_train.dump("data/y_" + str(taille_img_out) + "_" + str(it) + ".dat")
