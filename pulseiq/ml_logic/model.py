import os
from typing import Optional

import joblib
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

# Kolumnordningen MÅSTE matcha träningen exakt
FEATURE_ORDER = [
    "age", "gender", "pulse_rate", "systolic_bp", "diastolic_bp",
    "glucose", "height", "weight", "bmi",
    "family_diabetes", "hypertensive", "family_hypertension",
    "cardiovascular_disease", "stroke",
]


class DiabetesModel:
    """Kapslar in modellen. Utsidan ser bara predict()."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Modellfil saknas: {model_path}")
        self._model = joblib.load(model_path)  # privat – läcker aldrig ut

    def predict(self, features: dict) -> dict:
        X = pd.DataFrame([features])[FEATURE_ORDER]
        prediction = self._model.predict(X)
        result = {"diabetic_prediction": str(prediction[0])}
        if hasattr(self._model, "predict_proba"):
            result["diabetic_risk"] = float(self._model.predict_proba(X)[0][1])
        return result


_model: Optional[DiabetesModel] = None


def load_model() -> None:
    """Anropas en gång vid API-startup."""
    global _model
    _model = DiabetesModel()


def get_model() -> DiabetesModel:
    if _model is None:
        raise RuntimeError("Modellen är inte laddad")
    return _model
