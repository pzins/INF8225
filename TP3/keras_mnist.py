# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
K.set_image_dim_ordering('th')
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train)
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype('float32')
# define data preparation
# datagen = ImageDataGenerator(rotation_range=90)
shift=0.2
# datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


# fit parameters from data
datagen.fit(X_train)

# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train):
	# create a grid of 3x3 images
	pyplot.imshow(X_batch[0].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	