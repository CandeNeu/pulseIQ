from pulseiq.ml_logic.data import preprocess, download_clean_data
from pulseiq.ml_logic.train import train_model
from xgboost import XGBClassifier
from pulseiq.params import *
from pulseiq.ml_logic.registry import *

def main():
    download_df = download_clean_data(DATABASE_PATH)
    df, X, y_diabetic, y_hypertensive, y_cv = preprocess(download_df)

    model_diabetes = train_model(XGBClassifier, X, y_diabetic, CONFIG_DIABETES)
    model_hypertension = train_model(XGBClassifier, X, y_hypertensive, CONFIG_HYPERTENSION)
    model_cv = train_model(XGBClassifier, X, y_cv, CONFIG_CV)

    save_model(model_diabetes, 'diabetes')
    save_model(model_hypertension, 'hypertension')
    save_model(model_cv, 'cv')






if __name__ == '__main__':
    main()
