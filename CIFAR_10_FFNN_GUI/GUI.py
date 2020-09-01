import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical as tcg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


model = Sequential([
  Flatten(input_shape=(32, 32, 3)),
  Dense(256, activation='relu'),
  Dense(128, activation='relu', kernel_initializer='he_uniform'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
  
])

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.load_weights('FFNN_CIFAR10.h5')

st.markdown("<h1 style='text-align: center; color: black;'>CIFAR-10 Image Recognizer using FFNN</h1>", unsafe_allow_html=True)

uploaded_file_sample = st.file_uploader("Upload a sample image", type=["png", "jpg", "jpeg"])


SIZE = 192


st.set_option('deprecation.showfileUploaderEncoding', False)


if uploaded_file_sample is not None:
#data = uploaded_file
    import PIL  
    from PIL import Image  

    
    # creating a image object (main image)  
    im1 = Image.open(uploaded_file_sample)  
    
    # save a image using extension 
    im1 = im1.save("sample.jpg") 

    image = cv2.imread('sample.jpg',1)




    img = cv2.resize(image, (32, 32))

    img = np.expand_dims(img, 0)

    st.write('Model Input - Sample Image')
    st.image(image,use_column_width=True)

uploaded_file_test = st.file_uploader("Upload a test image", type=["png", "jpg", "jpeg"])

if uploaded_file_test is not None:

    
    # creating a image object (main image)  
    im2 = Image.open(uploaded_file_test)  
    
    # save a image using extension 
    im2 = im2.save("test.jpg") 

    imaget = cv2.imread('test.jpg',1)




    img2 = cv2.resize(imaget, (32, 32))

    img2 = np.expand_dims(img2, 0)

    st.write('Model Input - Test Image')
    st.image(imaget,use_column_width=True)


if st.button('Predict'):

    prediction = model.predict(img)

    predict = str(model.predict_classes(img))

    st.write('Predicted Class for Sample Image: ',predict)

    predictiont = model.predict(img2)

    predictt = str(model.predict_classes(img2))

    st.write('Predicted Class for Test Image: ',predictt)



        

    history=np.load('my_history.npy',allow_pickle='TRUE').item()

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    st.pyplot()

    plt.plot(history['val_loss'])
    plt.plot(history['val_accuracy'])
    plt.grid(True)
    plt.title('model loss curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    st.pyplot()

