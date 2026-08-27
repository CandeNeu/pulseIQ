from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np


def train_model(model_class, X, y, params):
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

    # Devolvés, por ejemplo, el modelo del último fold, o reentrenás con todo el dataset
    final_model = model_class(**params)
    final_model.fit(X, y)

    return final_model
