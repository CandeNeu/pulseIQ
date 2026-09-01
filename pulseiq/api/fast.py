from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from pulseiq.api.schemas import PatientFeatures
from pulseiq.ml_logic.registry import load_model, load_model_from_bucket
from pulseiq.params import *

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
    diabetic_model = app.state.diabetic
    data = pd.DataFrame(features.model_dump(), index=[0])
    diabetic_prediction = round(float(diabetic_model.predict_proba(data)[0][0]), 2)

    hypertensive_model = app.state.hypertensive

    hypertensive_prediction = round(float(hypertensive_model.predict_proba(data)[0][0]), 2)


    return {
        "diabetic": diabetic_prediction,
        "hypertensive": hypertensive_prediction,
    }
