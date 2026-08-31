import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from pulseiq.ml_logic.data import download_clean_data, preprocess
from pulseiq.ml_logic.oof_function import train_stacking_ensemble
from pulseiq.ml_logic.registry import save_model
from pulseiq.params import DATABASE_PATH


def predict_stacking_proba(base_models: dict, meta_model, X):
    """
    Genera las probabilidades predichas pasando primero por los 3 modelos base
    y luego por el Meta-XGBoost.
    """
    meta_features = [
        model.predict_proba(X)[:, 1] for _, model in base_models.items()
    ]
    X_meta = np.column_stack(meta_features)
    return meta_model.predict_proba(X_meta)[:, 1]


def evaluate_and_test(target_name: str, base_models: dict, meta_model, X_test, y_test):
    """
    Evalúa el ensemble final en el conjunto de prueba (Hold-out Test).
    """
    test_probs = predict_stacking_proba(base_models, meta_model, X_test)
    auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    print(f"\n==================== RESULTADOS: {target_name.upper()} ====================")
    print(f"ROC-AUC Test: {auc:.4f} | Brier Score: {brier:.4f}")

    # Muestra de 5 predicciones
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
    print("\nPrimeras 5 predicciones en pacientes de prueba:")
    for i in range(min(5, len(test_probs))):
        print(
            f"  Paciente {i+1} -> Probabilidad: {test_probs[i]:.4f} "
            f"({test_probs[i]*100:.1f}%) | Real: {int(y_test_arr[i])}"
        )

    return {"roc_auc": round(auc, 4), "brier_score": round(brier, 4)}


def main():
    # 1. Carga y preprocesamiento de datos
    print("⏳ Descargando y preprocesando datos...")
    download_df = download_clean_data(DATABASE_PATH)
    df, X, y_diabetic, y_hypertensive, y_cv = preprocess(download_df)

    # =========================================================================
    # [1/3] TARGET: DIABETES
    # =========================================================================
    print("\n" + "=" * 35 + " [1/3] TARGET: DIABETES " + "=" * 35)
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X, y_diabetic, test_size=0.2, random_state=42, stratify=y_diabetic
    )

    # Entrena los 3 modelos base con Optuna, genera OOF y entrena el Meta-XGBoost
    base_models_d, meta_d = train_stacking_ensemble(
        X=X_train_d,
        y=y_train_d,
        target_name="diabetes",
        n_trials_base=20,
        n_trials_meta=20,
    )

    metrics_d = evaluate_and_test("Diabetes", base_models_d, meta_d, X_test_d, y_test_d)
    save_model(
        model={"base_models": base_models_d, "meta_model": meta_d},
        params=metrics_d,
        model_name="ensemble",
        target="diabetes",
    )

    # =========================================================================
    # [2/3] TARGET: HYPERTENSION
    # =========================================================================
    print("\n" + "=" * 35 + " [2/3] TARGET: HYPERTENSION " + "=" * 35)
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X, y_hypertensive, test_size=0.2, random_state=42, stratify=y_hypertensive
    )

    base_models_h, meta_h = train_stacking_ensemble(
        X=X_train_h,
        y=y_train_h,
        target_name="hypertension",
        n_trials_base=20,
        n_trials_meta=20,
    )

    metrics_h = evaluate_and_test("Hypertension", base_models_h, meta_h, X_test_h, y_test_h)
    save_model(
        model={"base_models": base_models_h, "meta_model": meta_h},
        params=metrics_h,
        model_name="ensemble",
        target="hypertension",
    )

    # =========================================================================
    # [3/3] TARGET: CARDIOVASCULAR (CV)
    # =========================================================================
    print("\n" + "=" * 35 + " [3/3] TARGET: CARDIOVASCULAR " + "=" * 35)
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
        X, y_cv, test_size=0.2, random_state=42, stratify=y_cv
    )

    base_models_cv, meta_cv = train_stacking_ensemble(
        X=X_train_cv,
        y=y_train_cv,
        target_name="cv",
        n_trials_base=20,
        n_trials_meta=20,
    )

    metrics_cv = evaluate_and_test("Cardiovascular", base_models_cv, meta_cv, X_test_cv, y_test_cv)
    save_model(
        model={"base_models": base_models_cv, "meta_model": meta_cv},
        params=metrics_cv,
        model_name="ensemble",
        target="cv",
    )

    # =========================================================================
    # RESUMEN FINAL EN CONSOLA
    # =========================================================================
    print("\n" + "=" * 50)
    print("       🏆 RESUMEN ROC-AUC FINAL (HOLD-OUT TEST)")
    print("=" * 50)
    print(f" -> Diabetes:       ROC-AUC = {metrics_d['roc_auc']:.4f}")
    print(f" -> Hypertension:   ROC-AUC = {metrics_h['roc_auc']:.4f}")
    print(f" -> Cardiovascular: ROC-AUC = {metrics_cv['roc_auc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
