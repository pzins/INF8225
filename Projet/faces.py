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

def getAge(matlabDate):
	matlabDate = np.asscalar(matlabDate)
	python_datetime = datetime.fromordinal(int(matlabDate)) + timedelta(days=matlabDate%1) - timedelta(days = 366)
	return datetime.now().year - python_datetime.year

# group age into categories
# 0 => <20
# 1 => 20-30
# 2 => 30-40
# 3 => 40-50
# 4 => 50-60
# 5 => >60
def getAgeCategory(age):
	if age<20:
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
taille = 10000

a = np.array(d[0][0][2][:][0])

x_train = np.array([np.array(Image.open("/home/pierre/Downloads/wiki_crop/"+fname[0])) for fname in a[:taille]])


b = np.array(d[0][0][3][:][0][:taille])
"""
b = np.array(d[0][0][0][:][0][:taille])

for i in range(len(b)):
	b[i] = getAgeCategory(getAge(b[i]))
"""

# c = np.array(d[0][0][5][:][0][:taille])
# e = np.array(d[0][0][2][:][0][:taille])
# f = np.array(d[0][0][4][:][0][:taille])
# g = np.array(d[0][0][6][:][0][:taille])


"""
#version avec face location du dataset mais coo strange
counter = 0
for img in x_train:
	coo = c[counter][0]
	coo = coo.astype(int)
	x = coo[0]
	y = coo[1]
	x2 = coo[2]
	y2 = coo[3]
	print(coo)
	cv2.rectangle(img, (x,y), (x2,y2), (0,255,0), 2)

	# im = img[coo[1]:coo[3], coo[0]:coo[2]]

	cv2.imshow("Face found", img)
	cv2.waitKey(0)	
	counter += 1
"""
casc = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc)

real_set = []
label = []
counter = 0
for img in x_train:
	# print(counter)
	# img = cv2.imread(image)

	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img =cv2.equalizeHist(img )
	
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
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		im = img[y:y+h,x:x+w]
		# print(im.shape)
		if im.shape[0] < 128 or im.shape[1] < 128:
			break
		im = cv2.resize(im, (128, 128))
		real_set.append(im)
		test = 1
		print(counter)
		# print(b[counter])
		# cv2.imshow("Face found", im)
		# cv2.waitKey(0)

		# save img file		
		# name = "test/" + str(counter) + '.jpg'
		# scipy.misc.imsave(name, im)
		break
	if test == 1:
		if np.isnan(b[counter]):
			real_set.pop()
		else:
			label.append(b[counter])

	counter += 1
	# cv2.destroyAllWindows()

x_train = np.asarray(real_set)
y_train = np.array(label)
y_train = keras.utils.to_categorical(y_train, 2)

x_train = np.expand_dims(x_train, 4)

# print(x_train.shape)
x_train.dump("xtrain_128.dat")
y_train.dump("ytrain_128.dat")