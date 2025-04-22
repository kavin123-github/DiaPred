import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

st.markdown(
    """
    <div style='text-align: center; padding: 20px; border-radius: 15px; border: 4px solid gold; background: linear-gradient(to right, #fffde7, #e8f5e9);'>
        <h1 style='color: darkgreen;'>🩺 Luxury Diabetes Predictor</h1>
        <p style='font-size:18px;'>Enter your details below to predict your diabetes status.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input sliders or boxes
preg = st.number_input("Pregnancies (0–15):", min_value=0, max_value=15, step=1)
glucose = st.number_input("Glucose (50–200):", min_value=50, max_value=200, step=1)
bmi = st.number_input("BMI (15–50):", min_value=15.0, max_value=50.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function (0.0–2.5):", min_value=0.0, max_value=2.5, step=0.01)

if st.button("🔍 Predict"):
    features = np.array([[preg, glucose, bmi, dpf]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("You are likely Diabetic 🩺")
    else:
        st.success("You are NOT Diabetic ✅")
