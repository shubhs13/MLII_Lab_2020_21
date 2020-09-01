import streamlit as st
from PIL import Image
import classify
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown('<style>body{background-image: url("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"); background-repeat: no-repeat; background-attachment: fixed; background-size: 100% 100%;} body::before{content: ""; position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; background-color: rgba(1,1,1,0.80);}</style>',unsafe_allow_html=True)
st.markdown('<style>body{color: white; text-align: center;}</style>',unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")

classes = {
        0: '0 - Zero',
        1: '1 - One',
        2: '2 - Two',
        3: '3 - Three',
        4: '4 - Four',
        5: '5 - Five',
        6: '6 - Six',
        7: '7 - Seven',
        8: '8 - Eight',
        9: '9 - Nine'
      }

st.title("MNIST Classifier")
st.write("")
st.write("Train accuracy: 0.9973 \n\n Test accuracy: 0.9779")
st.write("")
st.subheader("By Jeevan Thukrul")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown('<style>body{color: white; text-align: center;}</style>', unsafe_allow_html=True)

        st.write("")

        if st.button('predict'):
                st.write("Result...")
                st.write("")
                label = classify.predict(uploaded_file)
                label = label.item()
                res = classes.get(label)
                st.title(res)
