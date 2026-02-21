
import streamlit as st
import numpy as np
import joblib
import os

# Load model and scaler safely
model_path = os.path.join(os.path.dirname(__file__), "solar_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)