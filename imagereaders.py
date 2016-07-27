import cv2
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from os import listdir
from os.path import isfile, join
import glob


# Dimensions of test images
row_length = 480 * 270


def read_images(folder):
    images = []
    mat = np.zeros((1, row_length), np.float32)
    folder = folder + "*/*/*/*"
    for filename in sorted(glob.glob(folder)):
        img2 = cv2.imread(os.path.join(folder, filename), 1)
        img = cv2.imread(os.path.join(folder, filename), 0)

        width, height = img.shape[:2]
        if img is not None:
            images.append(img2)
            outimg = np.reshape(img, (1, width * height))
            mat = np.vstack((mat, outimg))

    mat = np.delete(mat, 0, axis=0)
    return images, mat





def read_images_new(folder):
    images = []
    mat = np.zeros((1, row_length), np.float32)
    folder = folder + "*"
    for filename in sorted(glob.glob(folder)):
        img2 = cv2.imread(os.path.join(folder, filename), 1)
        img = cv2.imread(os.path.join(folder, filename), 0)
        #print os.path.join(folder, filename)
        width, height = img.shape[:2]


    mat = np.delete(mat, 0, axis=0)
    return images, mat




def read_answers(filename):
    returnlist = []
    with open(filename) as f:
        content = f.readlines()
        values = [x.split('\t') for x in content]
        for y in range(len(values)):
            num = int(values[y][1])
            for x in range(num):
                returnlist.append(np.float32(values[y][0]))
        return np.asarray(returnlist)


def read_answers_old(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip('\n') for x in content]
        #content = np.uint8(content)
        return np.asarray(content)



def read_video(filename, colorFlag=0):

    cap = cv2.VideoCapture(filename)
    outputlist = []
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        if colorFlag == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)

        outputlist.append(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    return np.asarray(outputlist)

def read_all_video(folder, colorFlag=0):
    output = []
    folder = os.getcwd() + folder
    folder = folder + "*"
    for filename in sorted(glob.glob(folder)):
        print filename
        output.append(read_video(filename))
    return (output)



def get_all_frames(folder, colorFlag=0):
    frames = []
    video = read_all_video(folder, colorFlag)
    for vid in video:
        length, width, height = vid.shape

        for x in range(length):
            frames.append(vid[x])


    return frames


def clipvideo(vid, start, end, fps, flow=0):
    retclip = []
    start *= fps
    end *= fps
    if flow == 0:
        end += 1
    for x in range(start, end):
        retclip.append(vid[x])
    return np.asarray(retclip)

def getFlowVid(vid,blockSize1, blockSize2):
    retval = []

    length, width, height = vid.shape
    winwid = width/blockSize1
    winhei = height/blockSize2
    for x in range(length-1):
        flow = cv2.calcOpticalFlowFarneback(vid[x], vid[x+1], None, 0.5, 2, 50, 3, 5, 1.2, 0)
        xflow = cv2.resize(flow[:,:,0], (blockSize1, blockSize2))
        yflow = cv2.resize(flow[:,:,1], (blockSize1, blockSize2))
        retval.append(xflow)
        retval.append(yflow)
        #print flow[0][1]
    return np.asarray(retval)

def getChannelsinVid(vid, colorFlags=0):
    retlist = []
    #length = len(vid.shape)
    #if length == 2:
    if colorFlags == 0:
        for x in range(1):
            retlist.append(vid)
    else:
        for x in range(3):
            retlist.append(vid)
    #else:
    #    wid, hei, chn = vid.shape
    #    for x in range(chn):
    #        for y in range(len(vid)):
    #            sam = []
    #            sam.append(vid[y])
    #        retlist.append(sam)

    return retlist

def makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0,  flow = 0):

    if colorFlag == 0:
        numFrames, wid, hei= video.shape
    else:
        numFrames, wid, hei, channels = video.shape
    normRate = normalFrameRate / desiredFrameRate
    secondsLong *= desiredFrameRate
    startsEverySecond *= desiredFrameRate
    returnList = []
    willtheRealReturnListPleaseStandUp = []
    for x in range(numFrames):
        if x % normRate == 0:
            returnList.append(video[x])
    if flow == 0:
        for x in range(len(returnList)):
            if (x % startsEverySecond == 0) and ((x+secondsLong + 1) < len(returnList)):
                clip = clipvideo(returnList, (x / desiredFrameRate), (x+secondsLong)/desiredFrameRate, desiredFrameRate, flow=flow)
                willtheRealReturnListPleaseStandUp.append(clip)
    else:
        for x in range(len(returnList)):
            if (x % startsEverySecond == 0) and ((x + secondsLong) < len(returnList)):
                clip = clipvideo(returnList, (x / desiredFrameRate), (x + secondsLong) / desiredFrameRate,
                                 desiredFrameRate, flow=flow)
                willtheRealReturnListPleaseStandUp.append(clip)

    print len(willtheRealReturnListPleaseStandUp)
    return willtheRealReturnListPleaseStandUp




def collectVidSegments(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0, flow = 0):
    folder = folder + "*/*.avi"
    returnList = []
    for filename in sorted(glob.glob(folder)):
        video = read_video(filename, colorFlag)

        lst = makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,colorFlag=colorFlag, flow=flow)

        for seg in lst:
            returnList.append(seg)

    return returnList


def collectAllVidSegments(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0, flow = 0):
    folder = folder + "*/*"
    returnList = []
    for filename in sorted(glob.glob(folder)):
        video = read_video(filename, colorFlag)
        print filename
        lst = makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,colorFlag=colorFlag, flow=flow)

        for seg in lst:
            returnList.append(seg)

    return returnList


def getIdentityFlow(video):
    retval = []

    length, width, height = np.asarray(video).shape
    for x in range(length - 1):
        flow = cv2.calcOpticalFlowFarneback(video[x], video[x + 1], None, 0.5, 2, 50, 3, 5, 1.2, 0)
        #xflow = cv2.resize(flow[:, :, 0], (50, 1))
        #yflow = cv2.resize(flow[:, :, 1], (50, 1))
        #retval.append(xflow)
        #retval.append(yflow)
        flw = cv2.resize(flow, (1, 50))
        if x == 0:
            returnarray = flw
        else:
            returnarray = np.hstack((returnarray, flw))


    return returnarray




def makeIdentitySegments(filename, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0):
    returnList = []
    video = read_video(filename)

    video = makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=colorFlag)

    for seg in video:
        leng, wid, hei = np.asarray(seg).shape
        newframes = []
        for x in range(leng):
            newframes.append(cv2.resize(seg[x], (5, 10)))
        flatflow = getIdentityFlow(newframes)

        returnList.append(flatflow)

    return returnList

def collectIdentitySegments(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0):
    folder = folder + "*/*"
    returnList = []
    for filename in sorted(glob.glob(folder)):


        lst = makeIdentitySegments(filename, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,
                              colorFlag=colorFlag)

        for seg in lst:
            returnList.append(seg)

    return returnList


def getIdentityFlow2(video):
    retval = []

    length, width, height = np.asarray(video).shape
    for x in range(length - 1):
        flow = cv2.calcOpticalFlowFarneback(video[x], video[x + 1], None, 0.5, 2, 50, 3, 5, 1.2, 0)
        flw = np.reshape(flow, (50, 1, 2))

        if x == 0:
            returnarray = flw
        else:
            returnarray = np.hstack((returnarray, flw))
    returnarray = np.reshape(returnarray, (2, 50, 60))
    return returnarray


def makeIdentitySegments2(filename, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0):
    returnList = []
    video = read_video(filename)

    video = makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=colorFlag)

    for seg in video:
        leng, wid, hei = np.asarray(seg).shape
        newframes = []
        for x in range(leng):
            newframes.append(cv2.resize(seg[x], (5, 10)))
        flatflow = getIdentityFlow2(newframes)

        returnList.append(flatflow)

    return returnList

def collectIdentitySegments2(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0):
    folder = folder + "*/*.avi"
    returnList = []
    for filename in sorted(glob.glob(folder)):

        lst = makeIdentitySegments2(filename, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,
                                   colorFlag=colorFlag)

        for seg in lst:
            returnList.append(seg)

    return returnList



def process_images(people, heights):
    i = 0
    imagelist = []
    imagestest = []
    imagestrain = []
    mattest = []
    mattrain = []
    for person in sorted(people):
        for height in sorted(heights):
            list, mat = read_images_new(
                person.getFileNames("/home/jessiefin/PycharmProjects/REUphase1/clean_video_frames/", height))
            imagelist.append(list)
            if i == 0:
                matrix = mat
            else:
                matrix = np.vstack((matrix, mat))

            if person.testortrain is "test":
                imagestest.append(list)
                mattest.append(mat)
            else:
                imagestrain.append(list)
                mattrain.append(mat)

            i += 1

    mattrain = np.asarray(mattrain)
    mattest = np.asarray(mattest)
    return imagelist, imagestrain, imagestest, mattrain, mattest

class Person:
    def __init__(self, number, height1, height2, height3, testortrain):
        self.number = number
        self.height1 = height1
        self.height2 = height2
        self.height3 = height3
        self.testortrain = testortrain



    def getCategories(self, height, bin1):
        bin3 = bin1
        bin25 = (height-75) / 25
        bin10 = (height-80) / 10
        return bin3, bin25, bin10

    def getHeights(self, num_classes):
        bin31, bin251, bin101 = self.getCategories(self.height1, 0)
        bin32, bin252, bin102 = self.getCategories(self.height2, 1)
        bin33, bin253, bin103 = self.getCategories(self.height3, 2)

        if num_classes is 3:
            return bin31, bin32, bin33
        elif num_classes is 5:
            return bin251, bin252, bin253
        else:
            return bin101, bin102, bin103


    def getFileNames(self, filename, heightgen):

        return filename  + self.testortrain + "/person" + str(self.number) + "/" + heightgen + "/"






if __name__ == "__main__":
    '''
    people = []

    heights = {"medium", "short", "tall"}


    person1 = Person(1, 93, 127, 165, "train")
    people.append(person1)
    person2 = Person(2, 82, 105, 151, "train")
    people.append(person2)
    person3 = Person(3, 94, 127, 165, "train")
    people.append(person3)
    person4 = Person(4, 114, 141, 188, "train")
    people.append(person4)
    person5 = Person(5, 99, 132, 173, "train")
    people.append(person5)
    person6 = Person(6, 97, 130, 173, "train")
    people.append(person6)
    person7 = Person(7, 99, 130, 173, "train")
    people.append(person7)
    person8 = Person(8, 104, 135, 178, "train")
    people.append(person8)
    person9 = Person(9, 112, 140, 180, "train")
    people.append(person9)
    person10 = Person(10, 99, 121, 166, "test")
    people.append(person10)

    imagelst, imagetrain, imagetest, flattrain, flattest = process_images(people, heights)
    '''

    lst = read_video('/home/jessiefin/PycharmProjects/REUphase1/videos/test/p2dyn_medium.mp4')
    test_videos = read_all_video('/videos/test/')
    print test_videos.shape
    print test_videos[2].shape