from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit
import random

import numpy as np

import theano
import theano.tensor as T


from logreg import LogisticRegression, load_data


from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from matplotlib import pyplot
from keras import backend as K

from scipy.ndimage import rotate
K.set_image_dim_ordering('th')

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from -2/n_in and 2/n_in
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-2 / n_in,
                    high=2 / n_in,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, nb_layer):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a relu activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.relu
        )

        self.hiddenLayers = [self.hiddenLayer]
        tmp = self.hiddenLayer.output
        for i in range(nb_layer-1):
            self.hiddenLayers.append(
            HiddenLayer(
                rng=rng,
                # input=self.hiddenLayers[-1].output,
                input=tmp,
                n_in=n_hidden,
                n_out=n_hidden,
                activation=T.nnet.relu
            ))
            tmp = self.hiddenLayers[-1].output

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden,
            n_out=n_out
        )
  

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum((i.W ** 2).sum() for i in self.hiddenLayers)
            + (self.logRegressionLayer.W ** 2).sum()
        )
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the n layer it is
        # made out of
        self.params=[]
        for i in self.hiddenLayers:
            self.params += i.params
        self.params += self.logRegressionLayer.params

        # keep track of model input
        self.input = input



  
def data_augmentation2(a):
    # print(type(setx))
    # print(setx.shape.eval())
    # a = setx.get_value()
    # b = setx.get_value()
    # print(a[0,:])
    X_train = a.get_value().reshape(a.get_value().shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')

    shift = 0.01
    datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift, rotation_range=5, horizontal_flip=False, vertical_flip=False)

    datagen.fit(X_train)
    # configure batch size and retrieve one batch of images
    for X_batch in datagen.flow(X_train, batch_size=a.get_value().shape[0]):
        a.set_value(X_batch.reshape(a.get_value().shape[0],784))
        break


def shift_up(mat, nb):
    for i in range(mat.shape[0]):
        if i >= mat.shape[0]-nb:
            mat[i,:].fill(0)
        else:
            mat[i,:] = mat[i+nb,:]
    return mat
def shift_right(mat, nb):
    for i in range(mat.shape[1]):
        if i >= mat.shape[1]-nb:
            mat[:,i].fill(0)
        else:
            mat[:,i] = mat[:,i+nb]
    return mat

def shift_down(mat, nb):
    for i in reversed(range(mat.shape[0])):
        if i < nb:
            mat[i,:].fill(0)
        else:
            mat[i,:] = mat[i-nb,:]
    return mat

def shift_left(mat, nb):
    for i in reversed(range(mat.shape[1])):
        if i < nb:
            mat[:,i].fill(0)
        else:
            mat[:,i] = mat[:,i-nb]
    return mat

def rotate_img(img, angle):
    return rotate(img, angle, reshape=False)

def data_augmentation(a):
    X_train = a.get_value().reshape(a.get_value().shape[0], 28, 28)
    for i in range(a.get_value().shape[0]):
        
        operation = random.randint(1,2)
        
        flip_dir = random.randint(0,1)
        angle = random.randint(0,4)
        shift = random.randint(0,2)
        shift_dir = random.randint(0,3)
        if not operation:
            if not flip_dir:
                X_train[i,:,:] = np.fliplr(X_train[i,:,:])
            else:
                X_train[i,:,:] = np.flipud(X_train[i,:,:])

        elif operation == 1:
            X_train[i,:,:] = rotate_img(X_train[i,:,:], angle)
        else:
            if shift_dir == 0:
                X_train[i,:,:] = shift_up(X_train[i,:,:], shift)
            elif shift_dir == 1:
                X_train[i,:,:] = shift_down(X_train[i,:,:], shift)
            elif shift_dir == 2:
                X_train[i,:,:] = shift_left(X_train[i,:,:], shift)
            else:
                X_train[i,:,:] = shift_right(X_train[i,:,:], shift)

    a.set_value(np.concatenate((a.get_value(),X_train.reshape(a.get_value().shape[0], 784)),0))



def test_mlp(learning_rate=0.01, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient


    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # print(train_set_x.shape.eval())
    # print(train_set_x.get_value().shape)
    # train_set_x.set_value(np.concatenate((train_set_x.get_value(),train_set_x.get_value()),0))
    # train_set_x.set_value(np.concatenate((train_set_x.get_value(),train_set_x.get_value()),0))
    
    data_augmentation(train_set_x)
    data_augmentation(train_set_x)
    
    train_set_y = T.concatenate((train_set_y, train_set_y), 0)
    train_set_y = T.concatenate((train_set_y, train_set_y), 0)
    # train_set_y.set_value(np.concatenate((train_set_y.get_value(),train_set_y.get_value()),0))
    # print(train_set_x.shape.eval())
    # print(train_set_x.get_value().shape)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    rng = np.random.RandomState(1234)
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        nb_layer = 2
    )
    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
       + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]




    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        # print(train_set_x[0,:].eval())
        # train_set_x, train_set_y = datasets[0]
        # data_augmentation(train_set_x)
        # print("DATA AUGMENTATION")
        # print(train_set_x[0,:].eval())

        for minibatch_index in range(n_train_batches):
            # learning_rate = epoch
            # print(minibatch_index)
            
            # print("-----------------------------------")
            # print(classifier.params[-2].eval()[0])
            # print("-----------------------------------")
            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # print("Iteration : %d" % iter)


            # check model on validation set
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            # if patience <= iter:
                # done_looping = True
                # break
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return best_validation_loss * 100, test_score * 100, epoch, (end_time - start_time)



if __name__ == '__main__':
    nb_neurones = [100]
    res = []
    for i in nb_neurones:
        best_validation_loss, test_score, epoch, duree = test_mlp(0.05, 0.0001, 100, 'mnist.pkl.gz', 50,  i)
        # best_validation_loss, test_score, epoch, duree = test_mlp(0.05, 0.0001, 100, 'mnist.pkl.gz', 200,  i)
        res.append([best_validation_loss, test_score, epoch, duree])
    for i in range(len(res)):
        print("%d: %f %%, %f %%, %d, %d" % (nb_neurones[i], res[i][0], res[i][1], res[i][2], res[i][3]))
    # test_mlp(0.05, 0.0001, 100, 'mnist.pkl.gz', 100, 30) pas mal si 1 hidden layer
    # params : learning_rate, L2_reg, n_epochs, dataset, batch_size, n_hidden



"""
10: 5.930000 %, 6.540000 %, 100, 82
50: 2.590000 %, 2.360000 %, 100, 101
100: 2.120000 %, 2.120000 %, 100, 74
500: 1.740000 %, 1.880000 %, 100, 123
1000: 1.710000 %, 1.720000 %, 100, 279
2000: 1.730000 %, 1.780000 %, 100, 566


2 hidden
100: 2.020000 %, 2.340000 %, 100, 91

3 hidden
100: 2.690000 %, 3.080000 %, 100, 109

4 hiddens
100: 89.360000 %, 88.650000 %, 100, 127

5 hiddens
100: 89.360000 %, 88.650000 %, 100, 147



data aug 2 hidden layer
100: 4.170000 %, 4.680000 %, 100, 183

200 neurones
100: 5.250000 %, 5.960000 %, 100, 130
400: 3.950000 %, 4.130000 %, 100, 318



**
100: 2.020000 %, 2.340000 %, 100, 94

"""