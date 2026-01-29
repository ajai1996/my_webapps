import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('lr.pkl', 'rb') as f:
    model= pickle.load(f)

with open('scaler.pkl', 'rb') as g:
    scaler= pickle.load(g)

st.title("Iris Species Prediction App")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=5.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=5.0)
predict=st.button("Predict Species")

if predict:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    species_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_dict[prediction[0]]
    
    st.success(f'The predicted species is: {predicted_species}')