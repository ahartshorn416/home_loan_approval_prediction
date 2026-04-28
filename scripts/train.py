"""
train.py

Author: Alison Hartshorn
Project: Home Loan Approval Prediction

Trains three models on the preprocessed HMDA data:
   1. Logisitic Regression (baseline)
   2. Random Forest
   3. XGBoost

Uses stratified K-fold cross-validation to select the best model.
Saves all models + CV results to models

"""
#-------------
# Imports
#-------------
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

#-------------
# Paths
#-------------
PROC_DIR   = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\data\\processed_data")
MODELS_DIR = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5


def load_data():
    print("[INFO] Loading processed training data...")
    X_train = joblib.load(PROC_DIR / "X_train.pkl")
    y_train = joblib.load(PROC_DIR / "y_train.pkl")
    print(f"       X_train: {X_train.shape}, y_train: {len(y_train)}")
    return X_train, y_train


def build_models() -> dict:
    """Return dict of model_name → unfitted model."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def cross_validate_models(models: dict, X: np.ndarray, y) -> pd.DataFrame:
    """Run stratified K-fold CV on all models, return summary DataFrame."""
    print(f"\n[INFO] Running {CV_FOLDS}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["accuracy", "f1_macro", "roc_auc", "precision_macro", "recall_macro"]
    results = []

    for name, model in models.items():
        print(f"\n  ▶ {name}")
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            scores = cv_results[key]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"]  = scores.std()
            print(f"    {metric:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
        results.append(row)

    return pd.DataFrame(results)


def select_best_model(cv_df: pd.DataFrame) -> str:
    """Pick model with highest mean ROC-AUC."""
    best = cv_df.sort_values("roc_auc_mean", ascending=False).iloc[0]
    print(f"\n[INFO] Best model by ROC-AUC: {best['model']} ({best['roc_auc_mean']:.4f})")
    return best["model"]


def train_final_models(models: dict, X: np.ndarray, y) -> dict:
    """Refit all models on full training data."""
    print("\n[INFO] Fitting all models on full training set...")
    fitted = {}
    for name, model in models.items():
        print(f"  ▶ Fitting {name}...")
        model.fit(X, y)
        fitted[name] = model
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        print(f"     Saved → models/{name}.pkl")
    return fitted


def main():
    print("=" * 60)
    print("HMDA 2023 — Model Training")
    print("=" * 60)

    X_train, y_train = load_data()
    models = build_models()

    # Cross-validation
    cv_df = cross_validate_models(models, X_train, y_train)

    # Save CV results
    cv_path = MODELS_DIR / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\n[OK]  CV results saved → {cv_path}")

    # Print summary table
    print("\n── Cross-Validation Summary ─────────────────────────────────")
    display_cols = ["model", "accuracy_mean", "f1_macro_mean", "roc_auc_mean",
                    "precision_macro_mean", "recall_macro_mean"]
    print(cv_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("─" * 60)

    best_name = select_best_model(cv_df)
    joblib.dump(best_name, MODELS_DIR / "best_model_name.pkl")

    # Final fit
    fitted_models = train_final_models(models, X_train, y_train)

    print(f"\n[DONE] All models saved to {MODELS_DIR}/")
    print(f"       Best model: {best_name}")
    print("       Run evaluate.py next.\n")


if __name__ == "__main__":
    main()