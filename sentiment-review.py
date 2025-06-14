#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.layers import Normalization,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy,BinaryCrossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st



df = pd.read_csv("IMDB Dataset.csv")
print(df.head())

lable_encoder = LabelEncoder()
y_train = lable_encoder.fit_transform(df["sentiment"])
print(y_train[:10])

vec = CountVectorizer(max_features=5000, stop_words='english')
x_train = vec.fit_transform(df["review"]).toarray()


model = Sequential(
    [
        Dense(units=64,activation="relu"),
        Dense(units=32,activation="relu"),
        Dense(units=1,activation="sigmoid")
    ]
)

model.compile(
    loss = BinaryCrossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
)

model.fit(x_train,y_train,epochs=5)





st.title("Movie Review Sentiment Classifier")
with st.form("review-classifier"):
    review = st.text_input("Enter the review ")
    submit = st.form_submit_button("Submit")
    if submit:
        st.success("Submitted Succesfully !")
        x_test = [review]
        x_test_vec = vec.transform(x_test).toarray()
        prediction = model.predict(x_test_vec)

        if prediction[0][0] >= 0.5:
            st.write("Postivie Review")
        else:
            st.write("Negative Review")
