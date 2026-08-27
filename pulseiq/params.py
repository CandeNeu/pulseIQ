import os
from pathlib import Path

DATABASE_URL = 'https://data.mendeley.com/public-files/datasets/m8cgwxs9s6/files/4c109a9f-2462-4dce-b93c-5789168c5401/file_downloaded'
DATABASE_PATH = '/home/davig/code/pulseIQ/raw_data/DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh (1).csv'

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

CONFIG_DIABETES = 'config_diabetic.json'
CONFIG_HYPERTENSION = 'config_hypertensive.json'
CONFIG_CV = 'config_cv.json'

MODEL_DIABETES = 'XGBoost_diabetic.json'
MODEL_HYPERTENSION = 'XGBoost_hypertensive.json'
MODEL_CV = 'XGBoost_cv.json'
