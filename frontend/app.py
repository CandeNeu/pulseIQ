import os
import requests
import streamlit as st

st.set_page_config(page_title="PulseIQ – Diabetes Risk", page_icon="🩺", layout="centered")

# --- API-URL ---
# Lokalt läser den localhost. På GCP sätter du env-variabeln API_URL (se deploy-steget).
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

st.title("🩺 PulseIQ – Diabetes Risk Prediction")
st.markdown("Fyll i patientens uppgifter för att få en prediktion från modellen.")

# --- Inputs (BYT UT mot dina riktiga features!) ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Ålder", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Kön", ["Female", "Male", "Other"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
    hba1c = st.number_input("HbA1c-nivå", min_value=3.0, max_value=15.0, value=5.7)
with col2:
    glucose = st.number_input("Blodsocker (glucose)", min_value=50, max_value=400, value=120)
    hypertension = st.selectbox("Högt blodtryck", ["Nej", "Ja"])
    heart_disease = st.selectbox("Hjärtsjukdom", ["Nej", "Ja"])
    smoking = st.selectbox("Rökning", ["never", "former", "current", "not current"])

# --- Bygg parametrar (nycklarna MÅSTE matcha din API:s förväntade namn) ---
params = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    "hypertension": 1 if hypertension == "Ja" else 0,
    "heart_disease": 1 if heart_disease == "Ja" else 0,
    "smoking_history": smoking,
}

if st.button("Predicera risk", type="primary"):
    try:
        # Om din API använder GET med query-params (som taxifare):
        response = requests.get(API_URL, params=params)

        # Om din API istället använder POST med JSON, använd denna rad istället:
        # response = requests.post(API_URL, json=params)

        response.raise_for_status()
        result = response.json()

        st.success("Prediktion mottagen ✅")
        st.write(result)

        # Om din API t.ex. returnerar {"diabetes_risk": 0.87}, kan du visa snyggt:
        # risk = result.get("diabetes_risk")
        # if risk is not None:
        #     st.metric("Estimerad risk", f"{risk:.1%}")

    except requests.exceptions.RequestException as e:
        st.error(f"Kunde inte nå API:et: {e}")
        st.info(f"Kontrollerar URL: {API_URL}")
