import pandas as pd
import numpy as np
import cv2

joints_path = './data/joints/'
img_type = 'png'
img_size = 48
jnt_num = 3

def load_data():
    myfile = pd.read_csv('./train_subsampled.csv')
    myfile = np.array(myfile[['ID', 'Category']].values)
    myfile = np.random.permutation(myfile)
    split_line = int(209*0.8)
    y_train = myfile[:split_line]
    y_test = myfile[split_line:]

    X_train = np.zeros((jnt_num, len(y_train), img_size, img_size, 1))
    for i in range(len(y_train)):
        namae = str(y_train[i][0])
        for count in range(jnt_num):
            img = cv2.imread(joints_path+namae+'_'+str(count)+'.'+img_type, 0)
            X_train[count][i] = img.reshape((img_size, img_size, 1))

    X_test = np.zeros((jnt_num, len(y_test), img_size, img_size, 1))
    for j in range(len(y_test)):
        namae = str(y_test[j][0])
        for count in range(jnt_num):
            img = cv2.imread(joints_path+namae+'_'+str(count)+'.'+img_type, 0)
            X_test[count][j] = img.reshape((img_size, img_size, 1))

    return (X_train, np.delete(y_train,0,1)), (X_test, np.delete(y_test,0,1))
