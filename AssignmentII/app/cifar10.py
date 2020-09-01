import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2 as o


def model_predict_cnn(f):
    img = take_input(f)
    cnn_model = tf.keras.models.load_model('app\\models\\cifar_with_cnn.h5')
    digit = cnn_model.predict(img)
    # digit = digit*100
    n = np.argmax(digit)
    return label_cifar(n)

def model_predict_ffd(f):
    img = take_input(f)
    wcnn_model = tf.keras.models.load_model('app\\models\\cifar_widout_cnn.h5')
    digit = wcnn_model.predict(img)
    # digit = digit*100
    n = np.argmax(digit)
    return label_cifar(n)

def take_input(f):
    filename = f
    img = load_img(filename, target_size=(32, 32))


    #img = o.imread(filename)
    #img = Image.fromarray(np.ones((28, 28, 1), dtype=np.uint8))
    #img = ImageOps.invert(img)
    #img = o.resize(img, (32,32))
    
    #img = o.cvtColor(img, o.COLOR_BGR2GRAY)
    
   
    #img = o.bitwise_not(img)
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 32, 32, 3)
    
    #img = img.astype('float32')
    img = img / 255.0 
    #img = np.expand_dims(img, axis=0)
    return img

def label_cifar(n):
    list_of_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    word = list_of_labels[n]
    return word
