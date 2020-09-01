from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import ImageOps
import numpy as np

import streamlit as st

@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('mnist_2.hdf5')
        print('Model Loaded')
        return model


def predict(image):
        loaded_model = get_model()
        image = load_img(image, target_size=(28, 28), color_mode = "grayscale")
        image = ImageOps.invert(image)
        image = img_to_array(image)
        image = image/255.0
        image = np.reshape(image,[1,28,28,1])

        classes = loaded_model.predict_classes(image)

        return classes
