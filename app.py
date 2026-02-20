import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("solar_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Solar Power Prediction App")

# User Inputs
distance = st.number_input("Distance to Solar Noon", value=0.0)
temperature = st.number_input("Temperature", value=25.0)
wind_speed = st.number_input("Wind Speed", value=5.0)
sky_cover = st.number_input("Sky Cover", value=2.0)
visibility = st.number_input("Visibility", value=10.0)
humidity = st.number_input("Humidity", value=50.0)
avg_wind = st.number_input("Average Wind Speed", value=5.0)
pressure = st.number_input("Average Pressure", value=1013.0)

if st.button("Predict"):

    features = np.array([[distance, temperature, wind_speed,
                          sky_cover, visibility, humidity,
                          avg_wind, pressure]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"Predicted Solar Power: {prediction[0]:.2f}")