
import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Load Saved Model & Scaler
# ----------------------------
model = joblib.load("solar_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Solar Power Prediction", page_icon="☀️")

st.title("☀️ Solar Power Generation Prediction App")
st.write("Enter weather parameters to predict solar power output.")

# ----------------------------
# User Inputs
# ----------------------------
distance_to_noon = st.number_input("Distance to Solar Noon", value=0.0)
temperature = st.number_input("Temperature (°C)", value=25.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=5.0)
sky_cover = st.number_input("Sky Cover", value=2.0)
visibility = st.number_input("Visibility", value=10.0)
humidity = st.number_input("Humidity (%)", value=50.0)
avg_wind_speed = st.number_input("Average Wind Speed (km/h)", value=5.0)
pressure = st.number_input("Average Pressure (hPa)", value=1013.0)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict Solar Power"):

    # Arrange inputs in correct order
    input_data = np.array([[distance_to_noon,
                            temperature,
                            wind_speed,
                            sky_cover,
                            visibility,
                            humidity,
                            avg_wind_speed,
                            pressure]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Show result
    st.success(f"🔋 Predicted Solar Power Output: {prediction[0]:.2f}")