
import numpy as np
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import keras

from sklearn.cross_validation import train_test_split
import imagereaders
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD, rmsprop
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import networks




if __name__ == "__main__":


    samples = []

    #read in and process videos and annotations
    videos = imagereaders.collectVidSegments('/home/jessiefin/PycharmProjects/REUExtension/dogcentric/', 15, 15, 4, 2)
    allanswers = imagereaders.read_answers('/home/jessiefin/PycharmProjects/REUExtension/dogcentric2.txt')


    #preprocess and reshape video
    for vid in videos:
        vid = imagereaders.getFlowVid(vid, 32, 32)
        vid = imagereaders.getChannelsinVid(vid)
        samples.append(vid)


    #normalize data
    mean = np.asarray(samples).mean()
    samples -= mean
    sig = np.asarray(samples).std()
    samples /= sig




    x_train_list, x_test_list, y_train_list, y_test_list = train_test_split(samples, allanswers, test_size=0.10, random_state=68)


    y_train = np.asarray(y_train_list)
    y_test_orig = np.asarray(y_test_list)

    y_train = np_utils.to_categorical(np.uint8(y_train), nb_classes=4)
    y_test = np_utils.to_categorical(np.uint8(y_test_orig), nb_classes=4)


    samples = np.asarray(samples)
    print samples.shape



    # Test pretrained model
    print "About to do CNN stuff"
    model = networks.temporalNet()
    sgd = rmsprop()
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    hist = model.fit(np.asarray(x_train_list), y_train, nb_epoch=8, verbose=1)
    print "Fit complete"

    out = model.predict_classes(np.asarray(x_test_list))
    print "Predicted model"
    score = model.evaluate(np.asarray(x_test_list), y_test)

    print(out)
    print np.uint8(y_test_orig)

    print score
