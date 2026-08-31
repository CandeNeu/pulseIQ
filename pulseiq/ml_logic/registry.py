import json
import os
import pickle
from google.cloud import storage
from pulseiq.params import *

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, params: dict, model_name: str, target: str) -> None:
    """Guarda el modelo (.pkl) y sus hiperparámetros/métricas (.json)."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{target}.pkl")
    params_path = os.path.join(MODELS_DIR, f"params_{model_name}_{target}.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, default=str)

    print(f"✅ Artefactos guardados: [{model_name}_{target}] en {MODELS_DIR}/")


def load_model(model_name: str, target: str):
    """Carga el modelo entrenado (.pkl)."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{target}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at path: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_params(model_name: str, target: str) -> dict:
    """Carga los parámetros y métricas guardadas (.json)."""
    params_path = os.path.join(MODELS_DIR, f"params_{model_name}_{target}.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No params found at path: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_from_bucket(target):
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs())
    target_blob = [blob for blob in blobs if target in blob.name]
    blob = target_blob[0]
    model_path = os.path.join(MODELS_DIR, f"XGBoost_{target}.pkl")
    blob.download_to_filename(model_path)


if __name__ == "__main__":

    load_model_from_bucket(DIABETES_SYNTAXIS)
    load_model_from_bucket(HYPERTENSIVE_SYNTAXIS)
