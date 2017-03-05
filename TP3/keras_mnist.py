# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from scipy.ndimage import rotate
K.set_image_dim_ordering('th')
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype('float32')
# define data preparation

pyplot.subplot(2,1,1)
pyplot.imshow(X_train[0].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
pyplot.subplot(2,1,2)
pyplot.imshow(rotate(X_train[0].reshape(28, 28), 90,  reshape=False), cmap=pyplot.get_cmap('gray'))
pyplot.show()
quit()
# datagen = ImageDataGenerator(rotation_range=0)
# shift=0.2
# datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
print(X_train[0].shape)

# configure batch size and retrieve one batch of images
for X_batch in datagen.flow(X_train):
	print(X_batch[0] == X_train[0])	
	# break
	# create a grid of 3x3 images
	pyplot.imshow(X_batch[0].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	