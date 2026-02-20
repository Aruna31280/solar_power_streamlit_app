import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("solar_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌞 Solar Power Generation Prediction")

st.write("Enter weather details to predict solar power output.")

# Input fields
distance = st.number_input("Distance to Solar Noon", value=0.0)
temperature = st.number_input("Temperature (°C)", value=25)
wind_speed = st.number_input("Wind Speed (m/s)", value=5.0)
sky_cover = st.number_input("Sky Cover", value=2)
visibility = st.number_input("Visibility (km)", value=10.0)
humidity = st.number_input("Humidity (%)", value=50)
avg_wind = st.number_input("Average Wind Speed (Period)", value=5.0)
pressure = st.number_input("Average Pressure (Period)", value=1013.0)

# Prediction button
if st.button("Predict Solar Power"):

    # Arrange input in same order as training
    features = np.array([[distance, temperature, wind_speed,
                          sky_cover, visibility, humidity,
                          avg_wind, pressure]])

    # Apply scaling
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)

    st.success(f"Predicted Solar Power: {prediction[0]:.2f}")