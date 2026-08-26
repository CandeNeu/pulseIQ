from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, StratifiedKFold
import numpy as np

def train_model(model: model,
                X: X,
                y: y,
                params = params
                ):

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits()

    scores = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

    model_ = model(**params)

    model_.fit(X_train, y_train)

    return model_
