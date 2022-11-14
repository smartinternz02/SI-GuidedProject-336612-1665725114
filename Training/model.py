#Import ImageDataGenerator Library
import tensorflow as tf
#Import ImageDataGenerator Library
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

#Define the parameters /arguments for ImageDataGenerator class
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,
        rotation_range=180,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)
#Applying ImageDataGenerator functionality to trainset and testset.
#Applying ImageDataGenerator functionality to trainset
x_train=train_datagen.flow_from_directory(
    directory= r"D:\mini project\dataset\train_set",
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical')

#Applying ImageDataGenerator functionality to test set
x_test=test_datagen.flow_from_directory(
    directory= r"D:\mini project\dataset\test_set",
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical')
#Importing the model building libraries
'''Importing the model building libraries'''
#to define linear initializations import Sequential
from keras.models import Sequential
#To add layers import Dense
from keras.layers import Dense
# to create a convolution kernel import Convolution2D
from keras.layers import Convolution2D
# Adding Max pooling Layer
from keras.layers import MaxPooling2D
# Adding Flatten Layer
from keras.layers import Flatten
# Initializing the model
model=Sequential()
# Adding CNN layers
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),
                        activation='relu'))
# Adding Max pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Adding Flatten Layer
model.add(Flatten()) 
# Adding Hidden Layers
model.add(Dense( kernel_initializer='uniform',activation='relu',units=300))
# Adding 2nd hidden layer
model.add(Dense( kernel_initializer='uniform',activation='relu',units=100))
# Adding 3rd hidden layer
model.add(Dense( kernel_initializer='uniform',activation='relu',units=60))
# Adding output layer
model.add(Dense( kernel_initializer='uniform',activation='softmax',units=3))
# Configure the learning process
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=["accuracy"])
# Training the model
model.fit_generator(x_train,steps_per_epoch=31,
                    epochs=15,
                    validation_data=x_test,
                    validation_steps=11)
#save model 
model.save('alert.h5')
print(x_train.class_indices)
#Random image prediction
#import numpy library
import tensorflow as tf
import numpy as np
#import load_model method to load our saved model
from tensorflow.keras.models import load_model
#import image from keras.preprocessing
from tensorflow.keras.preprocessing import image
#loading our saved model file
model = load_model(r"C:\Users\sripa\alert.h5")
img = image.load_img(r"D:\mini project\dataset\train_set\domestic\domestic (32).jpg",target_size=(64,64))

x = image.img_to_array(img)
#expanding the shape of image to 4 dimensions
x = np.expand_dims(x,axis=0)
pred = model.predict(x)
pred