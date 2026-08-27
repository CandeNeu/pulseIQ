from xgboost import XGBClassifier
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, name):
    model_path = FRONTEND_DIR / f"XGBoost_{name}.json"
    config_path = FRONTEND_DIR / f"config_{name}.json"

    model.get_booster().save_model(str(model_path))

    config_str = model.get_booster().save_config()
    with open(config_path, "w") as f:
        f.write(config_str)

    return model_path, config_path

def load_model(name):
    model_path = FRONTEND_DIR / f"XGBoost_{name}.json"
    config_path = FRONTEND_DIR / f"config_{name}.json"

    model = XGBClassifier()
    model.load_model(str(model_path))

    with open(config_path) as f:
        model.get_booster().load_config(f.read())

    return model
