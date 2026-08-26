import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ladda modellen EN gång när API:et startar ---
# JUSTERA sökvägen till din faktiska modellfil (se ls models/)
app.state.model = joblib.load("models/model.joblib")


@app.get("/")
def root():
    return {"status": "ok", "message": "PulseIQ API is running"}


@app.get("/predict")
def predict(
    age: int,
    gender: str,
    pulse_rate: int,
    systolic_bp: int,
    diastolic_bp: int,
    glucose: float,
    height: float,
    weight: float,
    bmi: float,
    family_diabetes: int,
    hypertensive: int,
    family_hypertension: int,
    cardiovascular_disease: int,
    stroke: int,
):
    # Kolumnnamn och ordning MÅSTE matcha träningen exakt
    X_new = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "pulse_rate": pulse_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "glucose": glucose,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "family_diabetes": family_diabetes,
        "hypertensive": hypertensive,
        "family_hypertension": family_hypertension,
        "cardiovascular_disease": cardiovascular_disease,
        "stroke": stroke,
    }])

    model = app.state.model
    prediction = model.predict(X_new)

    result = {"diabetic_prediction": str(prediction[0])}

    # Om modellen stödjer sannolikheter, lägg till risk-procenten
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0][1]
        result["diabetic_risk"] = float(proba)

    return result
