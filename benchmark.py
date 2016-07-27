import cv2
import numpy as np
import imagereaders
from sklearn.cross_validation import train_test_split
import networks
from keras.optimizers import SGD, Adam, rmsprop, clip_norm, Adagrad, adadelta
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt



#read in video.  Change file path to reflect location of folder containing videos
vidseg = imagereaders.collectIdentitySegments2('/home/jessiefin/PycharmProjects/REUExtension/dogcentric/', 15, 15, 4, 2)
#read in annotations from .txt file
allanswers = imagereaders.read_answers('dogcentric2.txt')



#normalize data
mean = np.asarray(vidseg).mean()
vidseg -= mean
sig = np.asarray(vidseg).std()
vidseg /= sig


#shuffle all data into training and testing data
x_train_list, x_test_list, y_train_list, y_test_list = train_test_split(vidseg, allanswers, test_size=0.1, random_state=68)

#nb_classes should be set equal to the number of people in dataset
y_train = np_utils.to_categorical(np.uint8(y_train_list), nb_classes=4)
y_test = np_utils.to_categorical(np.uint8(y_test_list), nb_classes=4)

#preprocessing
x_train = []
for vid in x_train_list:
    vid = imagereaders.getChannelsinVid(vid)
    x_train.append(vid)

x_test = []
for vid in x_test_list:
    vid = imagereaders.getChannelsinVid(vid)
    x_test.append(vid)


#model initializaiton and compilation
model = networks.polegsNet()
opt = rmsprop()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(np.asarray(x_train), np.asarray(y_train), nb_epoch=10)
out = model.predict_classes(np.asarray(x_test))

#print results
print out
print np.uint8(y_test_list)
score = model.evaluate(np.asarray(x_test), np.asarray(y_test))
print score

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


