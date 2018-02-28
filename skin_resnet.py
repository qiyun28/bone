# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:45:38 2018

@author: Cheng Zhong Yao
"""
from __future__ import print_function

from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import numpy as np
import resnet

import pandas as pd
import numpy as np
import cv2
import os

global label
import pdb;pdb.set_trace();
label = pd.read_csv('./train_subsampled.csv')
label = label.loc[:,['image_name','label_type']].dropna(axis=0, how='all')
label['label_type'].astype('object')
dummies = pd.get_dummies(label['label_type']).rename(columns=lambda x: 'label_type_' + str(x))
label = pd.concat([label, dummies], axis=1).drop(['label_type'], axis = 1)
del dummies


#img = cv2.imread("D:\\CA\\Opencv\\DSC_1822.jpg")

def traverseDirByOSWalk(path):
    path = os.path.expanduser(path)
    file_list=[]
    for (dirname, subdir, subfile) in os.walk(path):
        for f in subfile:
            file_list.append(f)
    return file_list

#path = '../new_data/SkintechMD/chin/'
def read_img(path):
    global label
    file_list = pd.DataFrame(traverseDirByOSWalk(path),columns = ['image_name'])
    label = file_list.merge(label, how='left', on='image_name')
    imgs_chin = []
    i= 0
    for f in label.image_name:
        imgpath = path + f
    #    print(imgpath)
        img = cv2.imread(imgpath)
        img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
        imgs_chin.append(img)
        i = i + 1
        if (i%100 == 0):
            print(i)
    imgs_chin = np.array(imgs_chin)
    return imgs_chin

imgs_chin= read_img('../new_data/SkintechMD/chin/')
imgs_lcheek = read_img('../new_data/SkintechMD/lcheek/')
imgs_rcheek = read_img('../new_data/SkintechMD/rcheek/')

label2 =np.array(label.iloc[:,1:])
#################

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')
batch_size = 10
nb_classes = 3
nb_epoch = 100
data_augmentation = False
# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3


imgs_chin = imgs_chin.astype('float32')
imgs_lcheek = imgs_lcheek.astype('float32')
imgs_rcheek = imgs_rcheek.astype('float32')
split = len(imgs_chin)
X_train1, X_test1 = imgs_chin[:split], imgs_chin[split:]
X_train2, X_test2 = imgs_lcheek[:split], imgs_lcheek[split:]
X_train3, X_test3 = imgs_rcheek[:split], imgs_rcheek[split:]
Y_train, Y_test = label2[:split],label2[split:]

# subtract mean and normalize
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
mean_image1 = np.mean(X_train1, axis=0)
X_train1 -= mean_image1
X_test1 -= mean_image1
X_train1 /= 128.
X_test1 /= 128.

mean_image2 = np.mean(X_train2, axis=0)
X_train2 -= mean_image2
X_test2 -= mean_image2
X_train2 /= 128.
X_test2/= 128.

mean_image3 = np.mean(X_train3, axis=0)
X_train3 -= mean_image3
X_test3 -= mean_image3
X_train3 /= 128.
X_test3 /= 128.



#model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model = resnet.ResnetBuilder.build_multi_input((3,32,32), 3)
model.summary()
weights_file='./first_resnet.h5py'
if os.path.exists(weights_file):
    model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)


#opt = SGD(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Not using data augmentation.')
history = model.fit([X_train1,X_train2,X_train3], Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
#          validation_data=(X_test, Y_test),
          shuffle=True,
#          validation_split=0.25,
          callbacks=[lr_reducer,  model_checkpoint, csv_logger]
          )

#
#import matplotlib.pyplot as plt
## list all data in history
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model acc')
#plt.ylabel('acc')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
##df_predict = model.predict(df_train,batch_size=1,verbose=1)
#model.save('first_resnet.h5py')
#
#
#Y_actual = np.argmax(Y_train,axis=1)
#Y_pred = np.argmax(model.predict([X_train1,X_train2,X_train3]),axis=1)
##Y_pred = np_utils.to_categorical(Y_pred, nb_classes)
#
#from sklearn.metrics import confusion_matrix
#confusion_matrix(Y_actual, Y_pred)