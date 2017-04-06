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

import scipy.misc

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
# print(data.shape)
d = data["wiki"]
taille_img_out = 32
nb_categories = 2

a = np.array(d[0][0][2][:][0])

dataset_size = a.shape[0]
a = a[:60000]
print(dataset_size)

batchSize = 10000

for it in range(int(dataset_size/batchSize)):

	x_train = np.array([np.array(cv2.imread("/home/pierre/Downloads/wiki_crop/"+fname[0], 1)) for fname in a[it*batchSize:(it+1)*batchSize]])

	b = np.array(d[0][0][3][:][0][it*batchSize:(it+1)*batchSize])
	"""
	dob = np.array(d[0][0][0][:][0][it*batchSize:(it+1)*batchSize])
	photo_date = np.array(d[0][0][1][:][0][it*batchSize:(it+1)*batchSize])

	for i in range(len(dob)):
		# dob[i] = getAgeCategory(getAge(dob[i], photo_date[i]))
		dob[i] = getAge(dob[i], photo_date[i])
	b = dob
	"""

	# c = np.array(d[0][0][5][:][0][:taille])
	# e = np.array(d[0][0][2][:][0][:taille])
	# f = np.array(d[0][0][4][:][0][:taille])
	# g = np.array(d[0][0][6][:][0][:taille])


	casc = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(casc)

	real_set = []
	label = []
	counter = 0
	for img in x_train:
		# img = cv2.imread(image)
		
		# if len(img.shape) == 3:
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if len(img.shape) == 2:
			counter += 1
			continue
		# img =cv2.equalizeHist(img )
		faces = faceCascade.detectMultiScale(
			img,
			scaleFactor=1.1,
			minNeighbors=10,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)
		test = 0
		for (x,y,w,h) in faces:
			# print(w,h)
			# w = 46
			# h = 56
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
			print(counter)

			# cv2.imshow("Face found", im)
			# cv2.waitKey(0)

			# save img file	
			# if counter < 250:	
				# name = "test/" + str(counter) + "_" + str(int(b[counter])) + '.jpg'
				# scipy.misc.imsave(name, im)
			break
		if test == 1:
			if np.isnan(b[counter]) or b[counter] == -1:
				real_set.pop()
			else:
				label.append(b[counter])

		counter += 1

		# cv2.destroyAllWindows()

	x_train = np.asarray(real_set)
	y_train = np.array(label)
	y_train = keras.utils.to_categorical(y_train, nb_categories)
	# x_train = np.expand_dims(x_train, 4)

	x_train.dump("data1000/32_large/xtrain_" + str(taille_img_out) + "_" + str(it) + ".dat")
	y_train.dump("data1000/32_large/ytrain_" + str(taille_img_out) + "_" + str(it) + ".dat")
#data7 => test
#data6 => size 128 + colored
#data4 => size 250	
#data5 => size 128
#data3 => avc age regression
#data2 => avc age category