from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from pulseiq.ml_logic.model import load_model, get_model, DiabetesModel
from pulseiq.api.schemas import PatientFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()          # laddas en gång vid startup
    yield


app = FastAPI(title="PulseIQ API", lifespan=lifespan)

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


@app.get("/health")
def health():
    from pulseiq.ml_logic import model
    return {"model_loaded": model._model is not None}


@app.get("/predict")
def predict(
    features: PatientFeatures = Depends(),
    model: DiabetesModel = Depends(get_model),
):
    return model.predict(features.model_dump())
