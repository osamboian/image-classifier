import cv2 as cv
import numpy as numpy
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#create dataset for visualizing the data
(training_images, training_labels),(testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Get the training data and models   
#this is where the image recognition model is created and trained
model = models.Sequential()
#Convolutional latyer is used to isolate features from the images
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='relu'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

x = model.fit(training_images, training_labels, epochs=200, validation_data=(testing_images, testing_labels))

model.save('image_classifier1.h5', x)
