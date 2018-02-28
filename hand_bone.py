import pandas as pd
import numpy as np
import cv2

joints_path = './data/joints/'
img_type = 'png'
img_size = 48

def load_data():
    myfile = pd.read_csv('./train_subsampled.csv')
    myfile = np.array(myfile[['ID', 'Category']].values)
    myfile = np.random.permutation(myfile)
    split_line = int(209*0.8)
    y_train = myfile[:split_line]
    y_test = myfile[split_line:]

    X_train = np.zeros((len(y_train), img_size, img_size, 1))
    for i in range(len(y_train)):
        namae = str(y_train[i][0])
        img = cv2.imread(joints_path+namae+'_0.'+img_type, 0)
        X_train[i] = img.reshape((img_size, img_size, 1))

    X_test = np.zeros((len(y_test), img_size, img_size, 1))
    for j in range(len(y_test)):
        namae = str(y_test[j][0])
        img = cv2.imread(joints_path+namae+'_0.'+img_type, 0)
        X_test[j] = img.reshape((img_size, img_size, 1))

    return (X_train, np.delete(y_train,0,1)), (X_test, np.delete(y_test,0,1))
