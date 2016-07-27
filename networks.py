import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adamax, rmsprop, clip_norm, Adagrad
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization



def temporalNet(weights=None):
    model = Sequential()

    model.add(Convolution3D(30, 20, 17, 17, subsample=(4,2,2), input_shape=(1, 120,32,32)))
    model.add(Activation(LeakyReLU()))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(13, 2, 2), strides=(13,2, 2)))

    model.add(Reshape((60, 4, 4)))


    model.add(Convolution2D(100, 3, 3))
    model.add(Activation(LeakyReLU()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())


    model.add(Dense(400))
    model.add(Activation(LeakyReLU()))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Activation(LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))

    if weights:
        model.load_weights(weights)

    return model


def polegsNet(weights=None):
    model = Sequential()
    model.add(Convolution3D(128, 2, 50, 10, activation='relu', input_shape=(1, 2, 50, 60)))
    #model.add(Activation(ELU()))
    #model.add(BatchNormalization())
    model.add(Reshape((128, 1, 51)))
    model.add(MaxPooling2D(pool_size=(1,20), strides=(1,10)))
    model.add(Flatten())

    model.add(Dense(128, activation='sigmoid'))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))


    if weights:
        model.load_weights(weights)

    return model


