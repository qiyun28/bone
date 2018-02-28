import cv2
import numpy as np

data_type = "png"
csv_name = "joints.csv"
label_path = "./label/"
oriImg_path = "./train/"
joints_path = "./joints/"
pixel = 600

_jntcsv = np.loadtxt(csv_name, dtype='int', delimiter=",")

# joint box radius
js = 24
for fjnt in _jntcsv:
    namae = str(fjnt[0])
    oriImg = cv2.imread(oriImg_path+namae+'.'+data_type, 0)
    fjnt = np.delete(fjnt, 0).reshape(14,2)
    count = 0
    for jnt in fjnt:
        roi = oriImg[jnt[1]-js:jnt[1]+js, jnt[0]-js:jnt[0]+js]
        # box is out of bound
        if roi.size == 0:
            a = jnt[1]-js
            if a<0: a=0
            b = jnt[1]+js
            if b>pixel-1: b=pixel-1
            c = jnt[0]-js
            if c<0: c=0
            d = jnt[0]+js
            if d>pixel-1: d=pixel-1
            roi = np.array(oriImg[a:b, c:d])
            apad = (js*2-(b-a))/2
            cpad = (js*2-(d-c))/2
            roi = np.pad(roi, [(apad, (js*2-(b-a))-apad), (cpad, (js*2-(d-c))-cpad)], mode='constant', constant_values=0)

        cv2.imwrite(joints_path+namae+"_"+str(count)+'.'+data_type, roi)
        count += 1