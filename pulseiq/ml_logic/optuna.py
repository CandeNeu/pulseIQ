import catboost as cb
import numpy as np
import optuna
from pulseiq.ml_logic.registry import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


class TrainModel:

    def __init__(self, model: str, X, y, target: str, n_trials: int = 20):
        self.model_name = model.lower().strip()
        self.X = np.array(X)
        self.y = np.array(y)
        self.target = target
        self.n_trials = n_trials

        # Asignación automática según el modelo
        if self.model_name in ["xgb", "xgboost"]:
            self.estimator = xgb.XGBClassifier
            self.params = {
                "n_estimators": ("int", 100, 1000, 100),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float", 0.01, 0.3, True),
                "min_child_weight": ("int", 1, 10),
                "subsample": ("float", 0.5, 1.0),
                "colsample_bytree": ("float", 0.5, 1.0),
                "gamma": ("float", 1e-8, 1.0, True),
                "reg_alpha": ("float", 1e-8, 10.0, True),
                "reg_lambda": ("float", 1e-8, 10.0, True),
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }
        elif self.model_name in ["rf", "randomforest"]:
            self.estimator = RandomForestClassifier
            self.params = {
                "n_estimators": ("int", 100, 1000, 100),
                "max_depth": ("int", 3, 25),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
                "max_features": ("categorical", ["sqrt", "log2", None]),
                "random_state": 42,
                "n_jobs": -1,
            }
        elif self.model_name in ["cb", "catboost"]:
            self.estimator = cb.CatBoostClassifier
            self.params = {
                "iterations": ("int", 100, 1000, 100),
                "depth": ("int", 4, 10),
                "learning_rate": ("float", 0.01, 0.3, True),
                "l2_leaf_reg": ("float", 1e-3, 10.0, True),
                "random_seed": 42,
                "verbose": 0,
                "thread_count": -1,
            }
        else:
            raise ValueError(f"Modelo '{self.model_name}' no soportado.")

    def _sample(self, trial):
        out = {}
        for k, v in self.params.items():
            if isinstance(v, tuple):
                if v[0] == "int":
                    out[k] = trial.suggest_int(
                        k, v[1], v[2], step=v[3] if len(v) > 3 else 1
                    )
                elif v[0] == "float":
                    out[k] = trial.suggest_float(
                        k, v[1], v[2], log=v[3] if len(v) > 3 else False
                    )
                elif v[0] == "categorical":
                    out[k] = trial.suggest_categorical(k, v[1])
            else:
                out[k] = v
        return out

    def _objective(self, trial):
        params = self._sample(trial)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, X_val = self.X[train_idx], self.X[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            model = self.estimator(**params)
            if self.model_name in ["xgb", "xgboost"]:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif self.model_name in ["cb", "catboost"]:
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            else:
                model.fit(X_tr, y_tr)

            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))

        return float(np.mean(scores))

    def tune_and_train(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials)

        best_params = self._sample(optuna.trial.FixedTrial(study.best_params))

        best_model = self.estimator(**best_params)
        if self.model_name in ["cb", "catboost"]:
            best_model.fit(self.X, self.y, verbose=0)
        else:
            best_model.fit(self.X, self.y)

        payload = {
            "model": self.model_name,
            "target": self.target,
            "best_roc_auc": round(study.best_value, 5),
            "params": best_params,
        }
        save_model(best_model, payload, self.model_name, self.target)

        return best_model
