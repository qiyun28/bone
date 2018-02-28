import cv2
import numpy as np
from operator import itemgetter

img = cv2.pyrDown(cv2.imread('hammer.png', cv2.IMREAD_UNCHANGED))
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# find contours and get the external one
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

flag = 1
# a minAreaRect in red and
for c in contours:
    # get the min area rect
    rect = cv2.minAreaRect(c)
    #import pdb; pdb.set_trace();
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    print("rect is: " + str(rect))
    print("box is: " + str(box))
    cv2.drawContours(img, [box], 0, (0, 0, 255))
    box = sorted(box, key = itemgetter(1))
    xUp = box[0][0]*0.5 + box[1][0]*0.5
    yUp = box[0][1]*0.5 + box[1][1]*0.5
    xDn = box[2][0]*0.5 + box[3][0]*0.5
    yDn = box[2][1]*0.5 + box[3][1]*0.5
    img = cv2.circle(img, (int(xUp),int(yUp)), 1, (0,255,0), 1)
    img = cv2.circle(img, (int(xDn),int(yDn)), 1, (255,0,0), 1)
    img = cv2.circle(img, (int(rect[0][0]),int(rect[0][1])), 1, (100,100,100),1)
    #if flag == 1: break;

print(len(contours))
# cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imwrite('contours.png',img)
