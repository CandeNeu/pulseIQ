from pulseiq.ml_logic.data import preprocess, download_clean_data
from pulseiq.ml_logic.train import train_model
from xgboost import XGBClassifier
from pulseiq.params import *

def main():
    download_df = download_clean_data(DATABASE_PATH)
    df, X_scaled, y_diabetic, y_hypertensive, y_cv = preprocess(download_df)

    model_diabetes, predict_diabetes = train_model(XGBClassifier, X_scaled, y_diabetic, CONFIG_DIABETES)
    model_hypertension, predict_hypertension = train_model(XGBClassifier, X_scaled, y_hypertensive, CONFIG_HYPERTENSION)
    model_cv, predict_cv = train_model(XGBClassifier, X_scaled, y_cv, CONFIG_CV)

    print(f' Diabetes  prob % {predict_diabetes.mean()*100}')
    print(f' hypertension  prob % {predict_hypertension.mean()*100}')
    print(f' CardioVascular Disease  prob % {predict_cv.mean()*100}')




if __name__ == '__main__':
    main()
