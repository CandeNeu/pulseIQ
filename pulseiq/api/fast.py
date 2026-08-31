from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from pulseiq.api.schemas import PatientFeatures
from pulseiq.ml_logic.registry import load_model

app = FastAPI()


app.state.diabetic = load_model("diabetic")
app.state.hypertensive = load_model("hypertensive")


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
    diabetic_prediction = int(diabetic_model.predict(data)[0])

    hypertensive_model = app.state.hypertensive

    hypertensive_prediction = int(hypertensive_model.predict(data)[0])

    cv_model = app.state.cv
    cv_prediction = int(cv_model.predict(data)[0])

    return {
        "diabetic": diabetic_prediction,
        "hypertensive": hypertensive_prediction,
        "cv": cv_prediction,
    }
