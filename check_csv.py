import argparse
import cv2

img_path = "./train/"
data_type = "png"


parser = argparse.ArgumentParser()
parser.add_argument("coord", type=str, help="image name and joint coordinates")
args = parser.parse_args()
joints = args.coord

joints = joints.split(',')
namae = joints.pop(0)

img = cv2.imread(img_path+namae+'.'+data_type, 0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


for j in range(len(joints)/2):
	img = cv2.circle(img, (int(joints[j*2]),int(joints[j*2+1])), 1, (0, 0+20*j, 255-20*j), j*2+2)

cv2.imwrite(namae + '_check.'+data_type, img)