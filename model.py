# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import csv
import random
import numpy as np
import cv2

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.


from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
import pandas as pd

data_df =pd.read_csv('my_data/driving_log.csv')
data_df.columns = ['center','left','right','steer','throttle','break','speed']

lines = []
header = True
camera_images = []
steering_angles = []
for i in range(len(data_df)):
    steering_center = data_df['steer'][i]
    sa_cor = 0.2
    steering_left = steering_center + sa_cor
    steering_right = steering_center - sa_cor
    img_name1 = data_df['center'][i].split('\\')[-1]
    img_name2 = data_df['left'][i].split('\\')[-1]
    img_name3 = data_df['right'][i].split('\\')[-1]
    path1 = 'my_data/IMG/' + img_name1 
    path2 = 'my_data/IMG/' + img_name2 
    path3 = 'my_data/IMG/' + img_name3 
    img_center = np.asarray(Image.open(path1))
    img_left = np.asarray(Image.open(path2))
    img_right = np.asarray(Image.open(path3))
    camera_images.extend([img_center, img_left, img_right])
    steering_angles.extend([steering_center, steering_left, steering_right])
    
# Visualizing some random images with their labels
fig, ax = plt.subplots(3,3, figsize=(16,8))
fig.subplots_adjust(hspace = .5, wspace=1)
ax = ax.ravel()
for i in range(0,8,3):
    #Creating a random idx number that will correspond to a left-cam image
    idx = random.randint(10, len(camera_images))
    idx = idx - (idx % 3) + 1 
    
    #Creating left, center, right images
    img_l = camera_images[idx]
    img_c = camera_images[idx - 1]
    img_r = camera_images[idx + 1 ]
    
    #Plotting images
    ax[i].imshow(img_l)
    ax[i].set_title('Left Camera')
    
    ax[i+1].imshow(img_c)
    ax[i+1].set_title('Center Camera')
    
    ax[i+2].imshow(img_r)
    ax[i+2].set_title('Right Camera')
    
augmented_imgs, augmented_sas= [],[]

for aug_img,aug_sa in zip(camera_images,steering_angles):
    augmented_imgs.append(aug_img)
    augmented_sas.append(aug_sa)
    
    #Flipping the image
    augmented_imgs.append(cv2.flip(aug_img,1))
    
    #Reversing the steering angle for the flipped image
    augmented_sas.append(aug_sa*-1.0) 

X_train, y_train = np.array(augmented_imgs), np.array(augmented_sas)
X_train, y_train = np.array(camera_images), np.array(steering_angles)


def preprocess(image):
    import tensorflow as tf
    #Resizing the image
    return tf.image.resize_images(image, (200, 66))


#resize
#Keras Sequential Model
model = Sequential()
#Image cropping to get rid of the irrelevant parts of the image (the hood and the sky)
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#Pre-Processing the image
model.add(Lambda(preprocess))
model.add(Lambda(lambda x: (x/ 127.0 - 1.0)))

#The layers
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())

model.compile(loss='mse',optimizer='adam') #adaptive moment estimation
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=6)
model.save('model.h5') 