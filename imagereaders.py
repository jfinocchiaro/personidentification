import cv2
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from os import listdir
from os.path import isfile, join
import glob


# Dimensions of test images
row_length = 480 * 270

#Reads individual frames from folder and returns list of images and flattened raw image vector for SVC training if needed
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




#Same as above, but reads directly in specified folder
def read_images_new(folder):
    images = []
    mat = np.zeros((1, row_length), np.float32)
    folder = folder + "*"
    for filename in sorted(glob.glob(folder)):
        img2 = cv2.imread(os.path.join(folder, filename), 1)
        img = cv2.imread(os.path.join(folder, filename), 0)
        width, height = img.shape[:2]


    mat = np.delete(mat, 0, axis=0)
    return images, mat



#Reads in answers from .txt file as in README
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

#Read answers from .txt file one line at a time
def read_answers_old(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip('\n') for x in content]
        #content = np.uint8(content)
        return np.asarray(content)


#Returns numpy array of all frames in a video
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

#Returns a list of all the videos in a folder
def read_all_video(folder, colorFlag=0):
    output = []
    folder = os.getcwd() + folder
    folder = folder + "*"
    for filename in sorted(glob.glob(folder)):
        print filename
        output.append(read_video(filename))
    return (output)


#returns all the frames in a folder
def get_all_frames(folder, colorFlag=0):
    frames = []
    video = read_all_video(folder, colorFlag)
    for vid in video:
        length, width, height = vid.shape

        for x in range(length):
            frames.append(vid[x])


    return frames

#Clips video into fixed length clips (flow=0 means Optical Flow is desired)
def clipvideo(vid, start, end, fps, flow=0):
    retclip = []
    start *= fps
    end *= fps
    if flow == 0:
        end += 1
    for x in range(start, end):
        retclip.append(vid[x])
    return np.asarray(retclip)

#Returns optical flow frames for a video in size blockSize1 x blockSize2 blocks
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

#preprocesses the video for network input
def getChannelsinVid(vid, colorFlags=0):
    retlist = []
    
    if colorFlags == 0:
        for x in range(1):
            retlist.append(vid)
    else:
        for x in range(3):
            retlist.append(vid)
    
    return retlist

#normalizes frame rate and returns a list of video segments that are normalized for each video
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



#returns all the video segments in a folder.  depending on depth of folder, change glob statement and .avi statement
def collectVidSegments(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0, flow = 0):
    folder = folder + "*/*.avi"
    returnList = []
    for filename in sorted(glob.glob(folder)):
        video = read_video(filename, colorFlag)

        lst = makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,colorFlag=colorFlag, flow=flow)

        for seg in lst:
            returnList.append(seg)

    return returnList

#Less specific
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

#DEPRECATED DO NOT USE
def getIdentityFlow(video):
    retval = []

    length, width, height = np.asarray(video).shape
    for x in range(length - 1):
        flow = cv2.calcOpticalFlowFarneback(video[x], video[x + 1], None, 0.5, 2, 50, 3, 5, 1.2, 0)
        
        flw = cv2.resize(flow, (1, 50))
        if x == 0:
            returnarray = flw
        else:
            returnarray = np.hstack((returnarray, flw))

    return returnarray



#DEPRECATED DO NOT USE
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

#DEPRECATED DO NOT USE
def collectIdentitySegments(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0):
    folder = folder + "*/*"
    returnList = []
    for filename in sorted(glob.glob(folder)):
        lst = makeIdentitySegments(filename, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,
                              colorFlag=colorFlag)

        for seg in lst:
            returnList.append(seg)

    return returnList

#returns reshaed optical flow for use in benchmark
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

#returns video segments for benchmark optical flow
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

#returns list of all video segments collected in folder
def collectIdentitySegments2(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0):
    folder = folder + "*/*.avi"
    returnList = []
    for filename in sorted(glob.glob(folder)):

        lst = makeIdentitySegments2(filename, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,
                                   colorFlag=colorFlag)

        for seg in lst:
            returnList.append(seg)

    return returnList
