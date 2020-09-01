import streamlit as st
import tensorflow as tf
from PIL import Image , ImageOps
import cv2 
import numpy as np

labels =  ['airplane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
  
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def lm():
  
  model = tf.keras.models.load_model('D:\Programming\Sem_7\ML_2\Models\CNN_CIFAR10_new.h5')
  return model

model = lm()

st.write("""
  # CNN Classifier for CIFAR10
""")
file = st.file_uploader("Upload",type=['jpeg','jpg','png'])

def import_and_predict(image_data,model):


  size=(32,32)
  img = ImageOps.fit(image_data,size,Image.ANTIALIAS)
  img = np.asarray(img)
  img = img[np.newaxis,...]
  l = np.argmax(model.predict(img))
  i  = labels[l]
  return i

if file is None:
  st.text("Select an Image")
else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  p = import_and_predict(image , model)
  # p = 'hello'
  st.success(str(p))
