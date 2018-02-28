import cv2
import numpy as np
from operator import itemgetter
import math

# finger - finger need to be completed, only tip is known
# bones - all bones. have to return array cos python2 cannot in-place sort with key
def findFinger(finger, bones):
    for m in range(3):
        if len(bones) == 0: break;
        bonny = finger[m]
        bones = sorted(bones, key = lambda x: math.sqrt(math.pow(math.fabs(bonny[4]-x[0]),2) + math.pow(math.fabs(bonny[5]-x[1]),2)))
        if bones[0][3] < bonny[3]: break;
        finger.append(bones.pop(0))
    return bones

# bonny - tip of known finger
# finger - tip of unknown finger, an empty array, results is appended
# bones - all bones. have to return array cos python2 cannot in-place sort with key
def nearestBoneMid(bonny,  bones, finger):
    bones = sorted(bones, key = lambda x: math.sqrt(math.pow(math.fabs(bonny[2]-x[2]),2) + math.pow(math.fabs(bonny[3]-x[3]),2)))
    finger.append(bones.pop(0))
    return bones

# helper function to check whether bones are correctly marked
def paintFingerColor(img, finger, color):
    for i in range(len(finger)):
        img = cv2.circle(img, (int(finger[i][2]),int(finger[i][3])), 1, color, 1)

# helper funciton to check whether joints are correctly marked
def paintJointsColor(img, joints, color):
    for i in range(len(joints)):
        img = cv2.circle(img, (int(joints[i][0]),int(joints[i][1])), 1, color, 1)

# find the mid pts of joints
# finger - the finger array
def findJoints(finger):
    joints = []
    for i in range(len(finger) - 1):
        x = (finger[i][4] + finger[i+1][0]) / 2
        y = (finger[i][5] + finger[i+1][1]) / 2
        joints.append([x,y])
    return joints


# ============ get contours
img = cv2.pyrDown(cv2.imread('2539.png', cv2.IMREAD_UNCHANGED))
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# find contours and get the external one
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

# ============ store 3 points for each piece of bone
bones = []
# a minAreaRect in red
for c in contours:
    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # draw a red rectangle
    cv2.drawContours(img, [box], 0, (0, 255, 0))
    box = sorted(box, key = itemgetter(1))
    xUp = box[0][0]*0.5 + box[1][0]*0.5
    yUp = box[0][1]*0.5 + box[1][1]*0.5
    xDn = box[2][0]*0.5 + box[3][0]*0.5
    yDn = box[2][1]*0.5 + box[3][1]*0.5
    #img = cv2.circle(img, (int(xUp),int(yUp)), 1, (0,255,0), 1)
    #img = cv2.circle(img, (int(xDn),int(yDn)), 1, (255,0,0), 1)
    #img = cv2.circle(img, (int(rect[0][0]),int(rect[0][1])), 1, (100,100,100),1)
    bone = [xUp, yUp, rect[0][0], rect[0][1], xDn, yDn]
    bones.append(bone)

print(len(bones))

# ============ mark bones positions, store in 5 arrays as 5 fingers
# highest point is mid finger tip
bones = sorted(bones, key = itemgetter(1))
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
    hand.append(edgeA); hand.append(evenA); hand.append(midfinger);
    hand.append(evenB); hand.append(edgeB);
else:
    #thumb = edgeB; little = edgeA; index = evenB; ring = evenA;
    hand.append(edgeB); hand.append(evenB); hand.append(midfinger);
    hand.append(evenA); hand.append(edgeA);

# for checking purpose
#paintFingerColor(img, thumb, (0,0,255))
#paintFingerColor(img, index, (0,255,0))
#paintFingerColor(img, midfinger, (255,0,0))
#paintFingerColor(img, ring, (0,255,255))
#paintFingerColor(img, little, (255,0,255))
for f in range(len(hand)):
    paintFingerColor(img, hands[f], (0, 0+50*f, 255-50*f))
cv2.imwrite('contours.png',img)

# ============ mark joint mid points
#tJnt = findJoints(thumb)
#iJnt = findJoints(index)
#mJnt = findJoints(midfinger)
#rJnt = findJoints(ring)
#lJnt = findJoints(little)
joints = []
# for checking purpose
#paintJointsColor(img, tJnt, (0,0,255))
#paintJointsColor(img, iJnt, (0,255,0))
#paintJointsColor(img, mJnt, (255,0,0))
#paintJointsColor(img, rJnt, (0,255,255))
#paintJointsColor(img, lJnt, (255,0,255))
#cv2.imwrite('contours.png',img)

# ============ save each joint as a pic


