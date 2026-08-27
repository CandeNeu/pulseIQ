from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pulseiq.params import *


def train_model(model_class, X, y, params):
    if isinstance(params, (str, Path)):
        config_file = FRONTEND_DIR / params if isinstance(params, str) and not Path(params).is_absolute() else params
        with open(config_file, "r") as f:
            params = json.load(f)

    skf = StratifiedKFold(n_splits=5)
    scores = []
    fitted_models = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        model_ = model_class(**params)
        model_.fit(X_train, y_train)

        preds = model_.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds)
        scores.append(score)
        fitted_models.append(model_)

        print(f"Fold {i}: AUC = {score:.4f}")

    print(f"Mean AUC: {np.mean(scores):.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    final_model = model_class(**params)
    final_model.fit(X_train, y_train)

    predict = final_model.predict(X_test)

    return final_model, predict
