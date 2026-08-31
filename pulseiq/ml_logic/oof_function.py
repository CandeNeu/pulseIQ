import catboost as cb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pulseiq.ml_logic.optuna import TrainModel


def oof_function(models: dict, X, y, n_splits: int = 5):
    """Generates Out-Of-Fold probability predictions using the best base models."""
    X_vals = X.values if hasattr(X, "values") else np.array(X)
    y_vals = y.values if hasattr(y, "values") else np.array(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_features = np.zeros((len(y_vals), len(models)))

    for col_idx, (model_name, model_instance) in enumerate(models.items()):
        print(f"Generando OOF para: {model_name}...")

        for train_idx, val_idx in skf.split(X_vals, y_vals):
            X_train, y_train = X_vals[train_idx], y_vals[train_idx]
            X_val = X_vals[val_idx]

            # Instanciar copia con los mejores hiperparámetros
            clf = model_instance.__class__(**model_instance.get_params())

            if isinstance(clf, cb.CatBoostClassifier):
                clf.set_params(verbose=0)

            clf.fit(X_train, y_train)
            oof_features[val_idx, col_idx] = clf.predict_proba(X_val)[:, 1]

    return oof_features


def train_stacking_ensemble(
    X, y, target_name: str, n_trials_base: int = 20, n_trials_meta: int = 20
):
    """1. Optimiza y entrena los 3 modelos base con Optuna (XGBoost, RF, CatBoost).

    2. Genera matriz OOF de probabilidades.
    3. Optimiza y entrena el Meta-XGBoost con Optuna sobre las features OOF.
    """
    model_types = ["xgboost", "randomforest", "catboost"]
    tuned_base_models = {}

    # 1. Optimizar y entrenar cada modelo base con Optuna (se guardan individualmente)
    for model_type in model_types:
        print(
            f"\n[Optuna Base] Optimizando {model_type} para target '{target_name}'..."
        )
        trainer = TrainModel(
            model=model_type,
            X=X,
            y=y,
            target=target_name,
            n_trials=n_trials_base,
        )
        tuned_base_models[model_type] = trainer.tune_and_train()

    # 2. Generar matriz OOF (n_samples, 3 columnas de probabilidades)
    X_meta_oof = oof_function(tuned_base_models, X, y, n_splits=5)
    y_vals = y.values if hasattr(y, "values") else np.array(y)

    # 3. Optuna para el Meta-XGBoost sobre X_meta_oof
    print(
        f"\n[Optuna Meta] Optimizando Meta-XGBoost para '{target_name}' ({n_trials_meta} trials)..."
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def meta_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-6, 5.0, log=True
            ),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        cv_scores = []
        for train_idx, val_idx in skf.split(X_meta_oof, y_vals):
            X_m_tr, y_m_tr = X_meta_oof[train_idx], y_vals[train_idx]
            X_m_val, y_m_val = X_meta_oof[val_idx], y_vals[val_idx]

            meta_clf = XGBClassifier(**params)
            meta_clf.fit(X_m_tr, y_m_tr)
            preds = meta_clf.predict_proba(X_m_val)[:, 1]
            cv_scores.append(roc_auc_score(y_m_val, preds))

        return float(np.mean(cv_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(meta_objective, n_trials=n_trials_meta)

    # 4. Entrenar Meta-XGBoost final con mejores parámetros
    best_meta_params = study.best_params
    best_meta_params.update(
        {"eval_metric": "logloss", "random_state": 42, "n_jobs": -1}
    )

    meta_model = XGBClassifier(**best_meta_params)
    meta_model.fit(X_meta_oof, y_vals)

    print(
        f"   [OK] Meta-XGBoost final entrenado | Mejor CV ROC-AUC: {study.best_value:.4f}"
    )

    return tuned_base_models, meta_model
