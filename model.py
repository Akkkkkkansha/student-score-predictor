import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load best trained model
with open("best_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("📚 Student Score Predictor")
st.write("Enter the student details below to predict the final score:")

# Input fields
hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
attendance = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=80)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

if st.button("🔍 Predict Score"):
    # Prepare input DataFrame
    input_df = pd.DataFrame({
        "hours_studied": [hours],
        "attendance_rate": [attendance],
        "previous_score": [previous_score]
    })

    # Prediction
    predicted_score = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Score: {predicted_score:.2f} / 100")

    # Score band interpretation
    if predicted_score >= 90:
        st.info("Grade: A")
    elif predicted_score >= 80:
        st.info("Grade: B")
    elif predicted_score >= 70:
        st.info("Grade: C")
    else:
        st.info("Grade: D")

st.markdown("---")
st.caption("Developed by Akansha — Powered by Machine Learning & Streamlit")
