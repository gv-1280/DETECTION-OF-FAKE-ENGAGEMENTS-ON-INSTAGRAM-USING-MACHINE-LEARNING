# This file is part of the Instagram Engagement Checker.
# Copyright (c) 2025 Gautam Vaishnav
# Licensed under the MIT License. See LICENSE file for details.

#importing required libraries
import streamlit as st
import joblib
import numpy as np
import os  

#load the model
model = joblib.load('models/ensemble_model.pkl')

#app title 
st.title("📊 Instagram Engagement Authenticity Checker")

#app header
st.header("Enter Instagram Profile Stats")

#input fields 
followers = st.number_input("Number Of Followers ",min_value=1)
likes = st.number_input("Number Of Likes ",min_value=0)
comments = st.number_input("Number Of Comments",min_value=0)
average_comment_length = st.number_input("Avg. Comment Length",min_value=0)
emoji_comment_ratio = st.slider("Emoji Comment Ratio (%) ", min_value=0.0, max_value=1.0, step=0.01)
like_comment_ratio = st.slider("Like to Comment Ratio (%) ", min_value=0.0, max_value=1.0, step=0.01)
#engagement rate calculation 
engagement_rate = (likes + comments)/followers
st.markdown(f"** Engagement Rate **`{engagement_rate:.3f}`")

#Button trigger prediction 
if st.button("Predict Authenticity"):
    input_data = np.array([[average_comment_length,emoji_comment_ratio,like_comment_ratio,likes,followers,comments]])

    prediction = model.predict(input_data)

    #display result
    if prediction == 0:
        st.success("✅ This profile appears to be **Genuine**.")
    else:
        st.error("❌ This profile appears to be **Fake**.")