import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras import backend as k
from keras.layers import Flatten, Dense

from keras.models import load_model
import streamlit as st
from PIL import Image

st.title("HandWritten Digit Recognition")

model = load_model("handwritten digit recognition.h5")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

#----------------------------------------------------------------------------------

def reshapping(image):
  image = image.resize((28, 28))
  image = image.convert("L")
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = image.reshape((1, 28, 28, 1))
  image = image.astype("float32")
  image /= 255.0
  return image

def showimg(image):
  img = tf.squeeze(image).numpy()
  plt.imshow(img,cmap='gray')
  plt.show()

def prediction(image):
  r=model.predict(image)
  return r.argmax()

#------------------------------------------------------------------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image=reshapping(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result=prediction(image)
    st.write("The handwritten Digit in the image is ",result)
else:
    st.write("Please upload an image file.")