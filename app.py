# import cv2
# import os
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import models,utils
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img,img_to_array
# from tensorflow.python.keras import utils

# #importing all the helper fxn from helper.py which we will create later

# import streamlit as st

# import os

# import matplotlib.pyplot as plt

# import seaborn as sns

# # getting the current path

# model = pickle.load(open('static/model.pkl', 'rb'))

# # loading the feature extractor model


# def predictor(img_path,uploaded_file): # here image is file name 

#     prediction=model.predict(img_path).save('static/images/prediction/'+uploaded_file.name) 
#     return  prediction
# 0

# sns.set_theme(style="darkgrid")

# sns.set()

# from PIL import Image

# st.title('Baseball cap Identifier')

# def save_uploaded_file(uploaded_file):

#     try:

#         with open(os.path.join('static/images/upload/',uploaded_file.name),'wb') as f:

#             f.write(uploaded_file.getbuffer())

#         return 1    

#     except:

#         return 0
# uploaded_file = st.file_uploader("Upload Image")

# # text over upload button "Upload Image"

# if uploaded_file is not None:

#     if save_uploaded_file(uploaded_file): 
#         # display the image
#         prediction = predictor(os.path.join('static/images/upload/',uploaded_file.name),uploaded_file)
#         st.image('static/images/prediction/'+uploaded_file.name)

#         # deleting uploaded saved picture after prediction

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

# getting the current path

model = pickle.load(open('static/model.pkl', 'rb'))

# loading the feature extractor model


def predictor(img_path,uploaded_file): # here image is file name 

    prediction=model.predict(img_path).save('static/images/prediction/'+uploaded_file.name) 
    return  prediction
0

sns.set_theme(style="darkgrid")

sns.set()

from PIL import Image

st.title('Baseball cap Identifier')

def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('static/images/upload/',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0
uploaded_file = st.file_uploader("Upload Image",accept_multiple_files=True,type="png,jpg,gif")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 
        # display the image
        prediction = predictor(os.path.join('static/images/upload/',uploaded_file.name),uploaded_file)
        st.image('static/images/prediction/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction
