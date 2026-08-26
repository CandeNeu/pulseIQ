import os
import requests
import streamlit as st

st.set_page_config(page_title="PulseIQ – Diabetes Risk", page_icon="🩺", layout="centered")

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

st.title("🩺 PulseIQ – Diabetes Risk Prediction")
st.markdown("Enter the patient's details to get a prediction from the model.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=42)
    gender = st.selectbox("Gender", ["Female", "Male"])
    pulse_rate = st.number_input("Pulse rate", min_value=30, max_value=200, value=66)
    systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=250, value=110)
    diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=73)
    glucose = st.number_input("Glucose", min_value=2.0, max_value=30.0, value=5.88)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.65)
with col2:
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.2)
    family_diabetes = st.selectbox("Family history of diabetes", ["No", "Yes"])
    hypertensive = st.selectbox("Hypertensive", ["No", "Yes"])
    family_hypertension = st.selectbox("Family history of hypertension", ["No", "Yes"])
    cardiovascular_disease = st.selectbox("Cardiovascular disease", ["No", "Yes"])
    stroke = st.selectbox("Stroke", ["No", "Yes"])

# BMI beräknas automatiskt från längd och vikt
bmi = round(weight / (height ** 2), 2) if height > 0 else 0.0
st.metric("BMI (auto-calculated)", bmi)

params = {
    "age": age,
    "gender": gender,
    "pulse_rate": pulse_rate,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "glucose": glucose,
    "height": height,
    "weight": weight,
    "bmi": bmi,
    "family_diabetes": 1 if family_diabetes == "Yes" else 0,
    "hypertensive": 1 if hypertensive == "Yes" else 0,
    "family_hypertension": 1 if family_hypertension == "Yes" else 0,
    "cardiovascular_disease": 1 if cardiovascular_disease == "Yes" else 0,
    "stroke": 1 if stroke == "Yes" else 0,
}

if st.button("Predict risk", type="primary"):
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        result = response.json()

        pred = result.get("diabetic_prediction")
        if pred is not None:
            if str(pred).lower() in ("yes", "1"):
                st.error(f"Prediction: Diabetic")
            else:
                st.success(f"Prediction: Not diabetic")

        risk = result.get("diabetic_risk")
        if risk is not None:
            st.metric("Estimated risk", f"{risk:.1%}")

        with st.expander("Raw API response"):
            st.write(result)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        st.info(f"Checking URL: {API_URL}")
