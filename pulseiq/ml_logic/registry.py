from xgboost import XGBClassifier
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, name):
    model_path = MODELS_DIR / f"XGBoost_{name}.json"
    config_path = MODELS_DIR / f"config_{name}.json"

    model.get_booster().save_model(str(model_path))

    config_str = model.get_booster().save_config()
    with open(config_path, "w") as f:
        f.write(config_str)

    return model_path, config_path


def load_model(name):
    model_path = os.path.join(MODELS_DIR, f"XGBoost_{name}.json")
    config_path = os.path.join(MODELS_DIR, f"config_{name}.json")

    model = XGBClassifier()
    model.load_model(str(model_path))

    with open(config_path) as f:
        model.get_booster().load_config(f.read())

    return model
