import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# **Football Position Classification**
Are you physically fit to play the lineman football position?
This app predicts if you are or not!
Data obtained from [NCSA Sports](https://www.ncsasports.org/football/combine-results) 
""")

from PIL import Image
image = Image.open('footballpic.jpg')
st.image(image, use_column_width=True)

st.header('User Input Features')


def user_input_features():
    height = st.number_input("Enter player's height in inches")
    weight = st.number_input("Enter player's weight")
    forty_yard_dash = st.number_input("Enter player's 40 yard dash time")
    shuttle_run = st.number_input("Enter player's shuttle run time")
    three_cone = st.number_input("Enter player's three cone time")
    broad_jump = st.number_input("Enter player's broad jump in inches")
    vertical_jump = st.number_input("Enter player's vertical jump in inches")
    grade = st.number_input("Enter player's school grade")
    data = {'weight': weight,
            'forty_yard_dash': forty_yard_dash,
            'shuttle_run': shuttle_run,
            'three_cone': three_cone,
            'broad_jump': broad_jump,
            'vertical_jump': vertical_jump,
            'grade': grade,
            'height': height}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()



# Displays the user input features
st.subheader('User Input features')


st.write(input_df)

# Reads in saved classification model
def load_models():
    file_name = "Models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model
load_clf = load_models()
# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
if prediction == 0:
    st.write(':white_frowning_face: You do not have the physical requirements to be a lineman! :white_frowning_face: ')
if prediction == 1:
    st.write(':football: You are physically fit to be a lineman! :football:')



