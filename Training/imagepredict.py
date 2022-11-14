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
print(pred)