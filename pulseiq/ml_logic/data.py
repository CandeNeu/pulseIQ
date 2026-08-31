import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold

def download_clean_data(url):
    df = pd.read_csv(url)
        #Drop columns
    df1 = df.drop(columns=['systolic_bp',
                               'diastolic_bp',
                               'glucose',
                               'stroke'])

    return df1

def preprocess(df1: pd.DataFrame)-> pd.DataFrame:
    '''
    Cleaning the data and preprocessing
    '''

        #Binaryzing Diabetes to 0 = Normal and 1 = Diabetes

    map_diabetes = {
        'No': 0,
        'Yes': 1
    }

    df1['diabetic'] = df1['diabetic'].map(map_diabetes)


    # Binaryzing Sex to 0 = Female and 1 = Male


    map_sex = {
        'Female': 0,
        'Male': 1
    }

    df1['gender'] = df1['gender'].map(map_sex)


    scaler = StandardScaler()


    #scaling and getting our target

    X = df1[['age', 'pulse_rate', 'height', 'weight', 'bmi']]
   # X_scaled = scaler.set_output(transform='pandas').fit_transform(df1[['age', 'pulse_rate', 'height', 'weight', 'bmi']])
   # (XGB NO NECESITA SCALE; por eso!)
    y_diabetic = df1['diabetic']
    y_hypertensive = df1['hypertensive']
    y_cv = df1['cardiovascular_disease']

    return df1, X, y_diabetic, y_hypertensive, y_cv
