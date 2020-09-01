import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2 as o
from PIL import ImageOps, Image

def model_predict(f):
    img = take_input(f)
    mnist_model = tf.keras.models.load_model('app\\models\\mnist_model.h5')
    digit = mnist_model.predict(img)
    digit = digit*100
    return np.argmax(digit)

def take_input(f):
    filename = f
    #img = load_img(filename, grayscale=True, target_size=(28, 28))


    img = o.imread(filename)
    #img = Image.fromarray(np.ones((28, 28, 1), dtype=np.uint8))
    #img = ImageOps.invert(img)
    img = o.resize(img, (28,28))
    
    img = o.cvtColor(img, o.COLOR_BGR2GRAY)
    img = (255-img)
   
    #img = o.bitwise_not(img)
    # convert to array
    #img = img_to_array(img)
    # reshape into a single sample with 1 channel
    #img = img.reshape(1, 28, 28, 1)
    
    #img = img.astype('float32')
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

