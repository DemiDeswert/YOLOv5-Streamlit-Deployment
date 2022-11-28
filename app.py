import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils

#importing all the helper fxn from helper.py which we will create later

import streamlit as st

import os

import matplotlib.pyplot as plt

import seaborn as sns

current_path = os.getcwd()

# getting the current path

model = os.path.join(current_path, 'static\model.pkl')

# loading class_to_num_category



model = pickle.load(open('static/model.pkl', 'rb'))

# loading the feature extractor model


def predictor(img_path,uploaded_file): # here image is file name 

    img = load_img(img_path, target_size=(331,331))

    img = img_to_array(img)

    img = np.expand_dims(img,axis = 0)

    model.predict(img, confidence=40, overlap=30).save('static/images/prediction/',uploaded_file.name)
    prediction = Image.open('static/images/prediction/',uploaded_file.name)

    return(prediction)

sns.set_theme(style="darkgrid")

sns.set()

from PIL import Image

st.title('Dog Breed Classifier')

def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('static/images/upload/',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0
uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)

        prediction = predictor(os.path.join('static/images/upload/',uploaded_file.name),uploaded_file)

        os.remove('static/images/upload/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction
