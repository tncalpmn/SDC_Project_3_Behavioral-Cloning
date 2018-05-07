# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
import sklearn
import cv2
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL
from random import shuffle

#Globals
limitDataSample = -1 # -1 No limit
allImagesPathRoot = "./All_Images"
batch_size = 32
ep = 2

# Image Crop Parameters
top = 64
bottom = 22
left = 0
right = 0

#Original Image Dimentions
imageDim = (160, 320, 3)
orgDepth = imageDim[2]
orgHeight = imageDim[0]
orgWidth = imageDim[1]

allSamples = []

with open(allImagesPathRoot + "/" + "all_driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        allSamples.append(line)

print("CSV logs are read local...")

if limitDataSample is not -1:
    #shuffle(allSamples)
    allSamples = allSamples[:limitDataSample]
print("Amount of picture from each view: left, center, right:",len(allSamples)) # All Image Information can be found allSamples

allSamplesFilt = [allSamples[i][0:4] for i in range(len(allSamples))] # Get Rid of Unused Features
train_samples, validation_samples = train_test_split(allSamplesFilt, test_size=0.2) # Image Information (left,center,right,) of one frame
num_train_sample = len(train_samples)
num_validation_sample = len(validation_samples)


# NVIDIA Architecture + My improvements
def myNet(input_shape):
    # Create the Keras Model
    model = Sequential()

    # Crop Images
    model.add(Cropping2D(cropping=((top, bottom), (left, right)), input_shape=input_shape))

    # Normalization and Mean center
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))

    # Convolutional Layer 1
    model.add(Convolution2D(24, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    # Convolutional Layer 2
    model.add(Convolution2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    # Convolutional Layer 3
    model.add(Convolution2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    # Convolutional Layer 4
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    # Convolutional Layer 5
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))

    # Fully Connected Layer 2
    model.add(Dense(50))
    model.add(Dropout(0.5))  # TODO ? MAX Pooling?
    model.add(Activation('elu'))

    # Fully Connected Layer 3
    model.add(Dense(10))
    model.add(Activation('elu'))

    # Final Output
    model.add(Dense(1))

    return model

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = allImagesPathRoot + "/" + batch_sample[0] # 0 index is Center Image
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB);
                center_angle = float(batch_sample[3]) # 3 index is Steering
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

input_shape = (orgHeight,orgWidth,orgDepth) # Initial Input Image

print(input_shape)
model = myNet(input_shape)
model.compile(loss='mse',optimizer='adam')
historyObj = model.fit_generator(train_generator, steps_per_epoch=num_train_sample,
                        validation_data=validation_generator, validation_steps=num_validation_sample ,
                        epochs = ep, verbose=1)

model.save('model.h5')
print("Training over and model saved.")
