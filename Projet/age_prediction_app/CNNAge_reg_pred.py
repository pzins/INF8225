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

# load image file
files = ["pp.jpg"]
x_train = np.array([np.array(cv2.imread(fname, 1)) for fname in files])
x_train = x_train.astype('float32')
x_train /= 255

# load the trained model
model = load_model("model_age.h5")

# do the prediction
pred = model.predict(x_train)
print(pred)
