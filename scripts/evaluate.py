"""
evaluate.py

Author: Alison Hartshorn
Project: Home Loan Approval Prediction

Evaluates all trained models on the held-out test set.
Produces:
  - Classification reports
  - Confusion matrices
  - ROC curves (all models overlaid)
  - Precision-Recall curves
  - Feature importance plot (Random Forest + XGBoost)
  - Threshold tuning analysis
All plots saved to results/evaluation/.
"""
# ---------
# Imports
# ---------
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, RocCurveDisplay,
)

# ---------
# Paths
# ---------
PROC_DIR = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\data\\processed_data")
MODELS_DIR = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\models")
OUTPUT_DIR = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\results\\evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost"]
CLASS_LABELS = {0: "Denied", 1: "Approved"}
COLORS = {"LogisticRegression": "#2196F3", "RandomForest": "#4CAF50", "XGBoost": "#FF9800"}

sns.set_theme(style="whitegrid", palette="muted")


def load_artifacts():
    print("[INFO] Loading test data and models...")
    X_test = joblib.load(PROC_DIR / "X_test.pkl")
    y_test = joblib.load(PROC_DIR / "y_test.pkl")
    feat_names = joblib.load(PROC_DIR / "feature_names.pkl")
    best_name = joblib.load(MODELS_DIR / "best_model_name.pkl")

    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            print(f"[WARN] Model not found: {path}")

    print(f"       Loaded {len(models)} models, test size: {X_test.shape}")
    return X_test, y_test, feat_names, models, best_name

# -------------------------
# 1. Classification reports
# -------------------------
def print_classification_reports(models: dict, X_test, y_test):
    print("\n── Classification Reports (Test Set) ────────────────────────")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred,
                                       target_names=["Denied", "Approved"])
        print(f"\n  ── {name} ──")
        print(report)

# -------------------------
# 2. Confusion matrices
# -------------------------
def plot_confusion_matrices(models: dict, X_test, y_test):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm, annot=False, fmt="", cmap="Blues", ax=ax,
                    xticklabels=["Denied", "Approved"],
                    yticklabels=["Denied", "Approved"],
                    linewidths=0.5)

        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.4, f"{cm[i, j]:,}",
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if cm_pct[i, j] > 50 else "black")
                ax.text(j + 0.5, i + 0.65, f"({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_pct[i, j] > 50 else "black")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontweight="bold")

    plt.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "01_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK]  Saved → {path}")

# -------------------------
# 3. ROC curves
# -------------------------
def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS.get(name, "gray"), linewidth=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = OUTPUT_DIR / "02_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK]  Saved → {path}")

# ---------------------------
# 4. Precision-Recall curves
# ---------------------------
def plot_pr_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Baseline (AP = {baseline:.2f})")

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=COLORS.get(name, "gray"), linewidth=2,
                label=f"{name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = OUTPUT_DIR / "03_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK]  Saved → {path}")

# ---------------------------
# 5. Feature importance
# ---------------------------
def plot_feature_importance(models: dict, feat_names: list, top_n: int = 20):
    for model_name in ["XGBoost", "RandomForest"]:
        if model_name not in models:
            continue
        model = models[model_name]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            continue

        feat_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
        bars = ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1],
                       color=COLORS.get(model_name, "#607D8B"), alpha=0.85)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{model_name} — Top {top_n} Features", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
        plt.tight_layout()

        path = OUTPUT_DIR / f"04_feature_importance_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK]  Saved → {path}")

        # Print top 10
        print(f"\n  Top 10 features ({model_name}):")
        for _, row in feat_df.head(10).iterrows():
            print(f"    {row['feature']:45s} {row['importance']:.4f}")

# ---------------------------
# 6. Threshold tuning
# ---------------------------
def plot_threshold_tuning(models: dict, X_test, y_test, best_name: str):
    if best_name not in models:
        print(f"[SKIP] Best model {best_name} not found for threshold tuning")
        return

    model = models[best_name]
    if not hasattr(model, "predict_proba"):
        print(f"[SKIP] {best_name} has no predict_proba for threshold tuning")
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)

    precision_list, recall_list, f1_list, acc_list = [], [], [], []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        precision_list.append(precision_score(y_test, y_pred, zero_division=0))
        recall_list.append(recall_score(y_test, y_pred, zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
        acc_list.append(accuracy_score(y_test, y_pred))

    # Best threshold by F1
    best_idx = int(np.argmax(f1_list))
    best_thresh = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precision_list, label="Precision", color="#2196F3", linewidth=2)
    ax.plot(thresholds, recall_list, label="Recall", color="#F44336", linewidth=2)
    ax.plot(thresholds, f1_list, label="F1 Score", color="#4CAF50", linewidth=2)
    ax.plot(thresholds, acc_list, label="Accuracy", color="#FF9800", linewidth=2, linestyle="--")
    ax.axvline(best_thresh, color="black", linestyle=":", linewidth=1.5,
               label=f"Best F1 threshold = {best_thresh:.2f}")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Tuning — {best_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = OUTPUT_DIR / "05_threshold_tuning.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK]  Saved → {path}")
    print(f"\n  Optimal threshold (max F1): {best_thresh:.2f}")
    print(f"  At this threshold:")
    print(f"    Precision : {precision_list[best_idx]:.4f}")
    print(f"    Recall    : {recall_list[best_idx]:.4f}")
    print(f"    F1        : {f1_list[best_idx]:.4f}")
    print(f"    Accuracy  : {acc_list[best_idx]:.4f}")

    # Save optimal threshold
    joblib.dump(best_thresh, MODELS_DIR / "optimal_threshold.pkl")

# ---------------------------
# 7. Summary table
# ---------------------------
def print_summary_table(models: dict, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 (Denied)": f1_score(y_test, y_pred, pos_label=0),
            "F1 (Approved)": f1_score(y_test, y_pred, pos_label=1),
            "F1 Macro": f1_score(y_test, y_pred, average="macro"),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n── Final Model Comparison (Test Set) ────────────────────────")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("─" * 60)

# -------
# Main
# -------
def main():
    print("=" * 60)
    print("HMDA 2023 — Model Evaluation")
    print("=" * 60)

    X_test, y_test, feat_names, models, best_name = load_artifacts()

    print_classification_reports(models, X_test, y_test)
    print_summary_table(models, X_test, y_test)

    print("\n[INFO] Generating evaluation plots...")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_feature_importance(models, feat_names)
    plot_threshold_tuning(models, X_test, y_test, best_name)

    print(f"\n[DONE] All evaluation outputs saved to {OUTPUT_DIR}/")
    print(f"       Best model: {best_name}")
    print("       See outputs/evaluation/model_comparison.csv for full results.\n")


if __name__ == "__main__":
    main()