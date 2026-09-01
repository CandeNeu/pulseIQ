from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from pulseiq.api.schemas import PatientFeatures
from pulseiq.ml_logic.registry import load_model, load_model_from_bucket
from pulseiq.params import *
from pulseiq.ml_logic.data import preprocess_user_features

app = FastAPI()

load_model_from_bucket(DIABETES_SYNTAXIS)
load_model_from_bucket(HYPERTENSIVE_SYNTAXIS)

app.state.diabetic = load_model(DIABETES_SYNTAXIS)
app.state.hypertensive = load_model(HYPERTENSIVE_SYNTAXIS)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "PulseIQ API is running"}


@app.get("/predict")
def predict_diabetis(
    features: PatientFeatures = Depends(),
):
    data = pd.DataFrame(features.model_dump(), index=[0])

    diabetic_model = app.state.diabetic
    hypertensive_model = app.state.hypertensive

    # [0][1] = probability of the positive class ("at risk"), as a percentage
    diabetic_proba = round(float(diabetic_model.predict_proba(data)[0][1]) * 100, 1)
    hypertensive_proba = round(
        float(hypertensive_model.predict_proba(data)[0][1]) * 100, 1
    )

    return {
        "diabetic": int(diabetic_proba >= 50),  # 0/1 class
        "hypertensive": int(hypertensive_proba >= 50),
        "diabetic_proba": diabetic_proba,  # e.g. 73.4 (percent)
        "hypertensive_proba": hypertensive_proba,
    }
