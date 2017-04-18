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
import cv2
#CNN

files = ["test_data/p1.png",
		 "test_data/p2.png",
		 "test_data/p3.png",
		 "test_data/p4.png",
		 "test_data/p5.png",
		 "test_data/p6.png",
		 "test_data/p7.png",
		 "test_data/p8.png",
		 "test_data/p9.png",
		 "test_data/p10.png",
		 "test_data/p11.png",
		 "test_data/p12.png"
		 ]
x_train = np.array([np.array(cv2.imread(fname, 1)) for fname in files])
# x_train = np.expand_dims(x_train, axis=0)
print(x_train.shape)
epochs = 25
batch_size = 64
input_shape = (128, 128, 3)
data_augmentation = False

x_train = x_train.astype('float32')
x_train /= 255

model = load_model("age_regression.h5")

pred = model.predict(x_train)
print(pred)
