import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model=pickle.load(open('log_regression_project.pkl','rb'))

st.title("Model Deployment using Random Forest")


import pandas as pd
import streamlit as st

def user_input_parameters():

    distance_to_solar_noon = st.sidebar.number_input(
        "Enter distance-to-solar-noon value"
    )

    temperature = st.sidebar.number_input(
        "Enter temperature"
    )

    sky_cover = st.sidebar.number_input(
        "Enter sky-cover"
    )

    humidity = st.sidebar.number_input(
        "Enter humidity"
    )

    average_wind_speed_period = st.sidebar.number_input(
        "Enter average wind speed (period)"
    )

    # Keys must match trained model columns
    dict1 = {
        "distance-to-solar-noon": distance_to_solar_noon,
        "temperature": temperature,
        "sky-cover": sky_cover,
        "humidity": humidity,
        "average-wind-speed-(period)": average_wind_speed_period
    }

    features = pd.DataFrame(dict1, index=[0])
    return features


df = user_input_parameters()

button = st.button("Predict")

if button:
    pred = model.predict(df)

    st.subheader("Prediction")
    st.write("Predicted value is:", pred[0])
