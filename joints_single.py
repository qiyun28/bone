import cv2
import numpy as np
from operator import itemgetter
import math
import argparse
import sys
import glob

# some global constants
data_type = "png"
label_path = "./data/label/"
oriImg_path = "./data/train/"
checking_path = "./data/for_checking/"
joints_path = "./data/joints/"
pixel = 600
_jntcsv = []

# finger - finger need to be completed, only tip is known
# bones - all bones. have to return array cos python2 cannot in-place sort with key
def findFinger(finger, bones):
    for m in range(3):
        if len(bones) == 0: break;
        bonny = finger[m]
        bones = sorted(bones, key = lambda x: math.sqrt(math.pow(math.fabs(bonny[4]-x[0]),2) + math.pow(math.fabs(bonny[5]-x[1]),2)))
        # if next bone is higher than prev / next bone hor too far, stop
        if bones[0][3] < bonny[3] or math.fabs(bones[0][0] - bonny[4]) > 48: break;
        finger.append(bones.pop(0))
    return bones

# check whether the newly identified nearest tip bone is a tip without top bones
def isTip(arr):
    bonny = arr.pop(0)
    arr = sorted(arr, key = lambda x: math.sqrt(math.pow(math.fabs(bonny[2]-x[2]),2) + math.pow(math.fabs(bonny[3]-x[3]),2)))
    # if nearest bone is higher than tip, tip is wrong
    if arr[0][3] < bonny[3]:
        return 0
    else:
        return 1

# bonny - tip of known finger
# finger - tip of unknown finger, an empty array, results is appended
# bones - all bones. have to return array cos python2 cannot in-place sort with key
def nearestBoneMid(bonny,  bones, finger):
    bones = sorted(bones, key = lambda x: math.sqrt(math.pow(math.fabs(bonny[2]-x[2]),2) + math.pow(math.fabs(bonny[3]-x[3]),2)))
    # if bones[0] is not a finger tip
    while isTip(list(bones)) != 1:
        bones.append(bones.pop(0))
    finger.append(bones.pop(0))
    return bones

# helper function to check whether bones are correctly marked
def paintFingerColor(img, finger, color):
    for i in range(len(finger)):
        img = cv2.circle(img, (int(finger[i][2]),int(finger[i][3])), 1, color, i*3+3)

# helper funciton to check whether joints are correctly marked
def paintJointsColor(img, joints, color):
    for i in range(len(joints)):
        img = cv2.circle(img, (int(joints[i][0]),int(joints[i][1])), 1, color, i*3+3)

# find the mid pts of joints
# finger - the finger array
def findJoints(finger):
    joints = []
    for i in range(len(finger) - 1):
        x = (finger[i][4] + finger[i+1][0]) / 2
        y = (finger[i][5] + finger[i+1][1]) / 2
        joints.append([x,y])
    return joints

def computeJoints(namae):
    # ============ get contours
    # read image
    img = cv2.imread(label_path+namae+'.'+data_type, 0)
    oriImg = cv2.imread(oriImg_path+namae+'.'+data_type, 0)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    # ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)

    # ============ store 3 points for each piece of bone
    bones = []
    # a minAreaRect
    for c in contours:
        # get the min area rect
        rect = cv2.minAreaRect(c)
        # eliminate small contour dots
        if rect[1][0] < 4 or rect[1][1] < 4:
            continue
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # draw a rectangle
        cv2.drawContours(img, [box], 0, (255, 255, 0))
        # cv2.drawContours(oriImg, [box], 0, (255, 255, 0))
        # find the top 2 points to get top mid pt
        box = sorted(box, key = itemgetter(1))
        tip = box.pop(0)
        box = sorted(box, key = lambda x: math.sqrt(math.pow(math.fabs(tip[0]-x[0]),2) + math.pow(math.fabs(tip[1]-x[1]),2)))
        top = box.pop(0)
        xUp = tip[0]*0.5 + top[0]*0.5
        yUp = tip[1]*0.5 + top[1]*0.5
        # left with bottom 2 pts to get bottom mid pt
        xDn = box[0][0]*0.5 + box[1][0]*0.5
        yDn = box[0][1]*0.5 + box[1][1]*0.5
        bone = [xUp, yUp, rect[0][0], rect[0][1], xDn, yDn]
        bones.append(bone)

    if len(bones) != 19:
        print(namae + '  ----  contour error: ' + str(len(bones)))

    # ============ mark bones positions, store in 5 arrays as 5 fingers
    # highest point is mid finger tip
    bones = sorted(bones, key = itemgetter(3))
    midfinger = []; evenA = []; evenB = []; edgeA = []; edgeB = [];
    midfinger.append(bones.pop(0))

    bones = findFinger(midfinger, bones)
    bones = nearestBoneMid(midfinger[0], bones, evenA)
    bones = findFinger(evenA, bones)
    bones = nearestBoneMid(midfinger[0], bones, evenB)
    bones = findFinger(evenB, bones)

    bones = nearestBoneMid(evenA[0], bones, edgeA)
    bones = findFinger(edgeA, bones)
    bones = nearestBoneMid(evenB[0], bones, edgeB)
    bones = findFinger(edgeB, bones)

    # save fingers in a matrix. 0 is thumb
    hand = []
    if len(edgeA) == 3:
        #thumb = edgeA; little = edgeB; index = evenA; ring = evenB;
        hand.append(edgeA); hand.append(evenA); hand.append(midfinger); hand.append(evenB); hand.append(edgeB);
    else:
        #thumb = edgeB; little = edgeA; index = evenB; ring = evenA;
        hand.append(edgeB); hand.append(evenB); hand.append(midfinger); hand.append(evenA); hand.append(edgeA);

    # for checking purpose
    for f in range(len(hand)):
        paintFingerColor(img, hand[f], (0, 0+50*f, 255-50*f))
        # paintFingerColor(oriImg, hand[f], (0, 0+50*f, 255-50*f))
        if (f == 0 and len(hand[f]) != 3) or (f != 0 and len(hand[f]) != 4):
            print(namae + '  ----  bone tracking error')

    # ============ mark joint mid points
    joints = []
    for fin in hand:
        joints.append(findJoints(fin))

    tempArr=[]
    for fjnt in joints:
        for jnt in fjnt:
            tempArr.append(jnt)
    tempArr = np.hstack(tempArr).astype(int)
    print(np.array2string(tempArr, separator=','))
    return 0


if __name__ == "__main__":

    imgs = glob.glob(label_path+"*."+data_type)
    print('-'*30)
    print('Detecting joints...')
    print('-'*30)

    parser = argparse.ArgumentParser()
    parser.add_argument("namae", type=str, help="image name")
    args = parser.parse_args()
    imgname = args.namae

    computeJoints(imgname)



