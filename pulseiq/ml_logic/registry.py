from xgboost import XGBClassifier

import json
from pulseiq.params import *

import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, name):
    model_path = MODELS_DIR / f"XGBoost_{name}.json"
    config_path = MODELS_DIR / f"config_{name}.json"

    # 1. Save the trained binary/booster model
    model.save_model(str(model_path))

    # 2. Save the initialization hyperparameters dictionary
    params = model.get_params()
    with open(config_path, "w") as f:
        json.dump(params, f, indent=4)


def load_model(name):
    model_path = os.path.join(MODELS_DIR, f"XGBoost_{name}.json")
    config_path = os.path.join(MODELS_DIR, f"config_{name}.json")

    # 1. Load the hyperparameters to set up the wrapper
    with open(config_path, "r") as f:
        params = json.load(f)

    # 2. Instantiate with parameters and load trained weights
    model = XGBClassifier(**params)
    model.load_model(str(model_path))

    return model
