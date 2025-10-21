# src/train_model.py
"""
Train baseline (Logistic Regression) and XGBoost classifier safely.
Saves models, scaler, encoder, reports, and feature-importance plot.

Run from project root:
    python -m src.train_model
"""

import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Local imports
from src.data_loader import load_csv, validate_columns
from src.preprocessing import preprocess

# CONFIG
DATA_PATH = "data/health_insurance.csv"
MODELS_DIR = "models"
REPORT_DIR = "reports/models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_and_save(name, model, X_test, y_test, out_prefix):
    """Evaluate a model and save metrics + confusion matrix + ROC plot."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)
        else:
            y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = float("nan")

    cls_report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Save text report
    report_text = (
        f"Model: {name}\n"
        f"Time: {datetime.utcnow().isoformat()}Z\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"ROC AUC: {roc_auc:.4f}\n\n"
        f"Classification Report:\n{cls_report}\n\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    with open(f"{out_prefix}_report.txt", "w") as f:
        f.write(report_text)

    # Confusion matrix figure
    plt.figure(figsize=(5,4))
    plt.title(f"Confusion Matrix - {name}")
    sns_cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.imshow(sns_cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion.png")
    plt.clf()

    # ROC curve
    if not np.isnan(roc_auc):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0,1],[0,1],"--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_roc.png")
        plt.clf()

    print(f"[{name}] Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")
    return {"accuracy": acc, "roc_auc": roc_auc, "report_text": report_text}


def save_model_artifacts(out_path_base, **artifacts):
    """Save multiple artifacts (model, scaler, encoder, feature_list)."""
    joblib.dump(artifacts, out_path_base)
    print(f"Saved artifacts to {out_path_base}")


def main():
    # 1) Load data
    print("Loading data...")
    df = load_csv(DATA_PATH)
    validate_columns(df)

    # 2) Preprocess
    print("Preprocessing data...")
    df_proc, le_gender, scaler, feature_cols = preprocess(df, scale_numeric=True)

    X = df_proc[feature_cols].copy()
    y = df_proc["disease_risk"].astype(int).copy()

    # 3) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4) Logistic Regression baseline
    print("Training Logistic Regression baseline...")
    logreg = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    lr_out_prefix = os.path.join(REPORT_DIR, "logreg")
    os.makedirs(os.path.dirname(lr_out_prefix), exist_ok=True)
    lr_metrics = evaluate_and_save("LogisticRegression", logreg, X_test, y_test, lr_out_prefix)

    logreg_artifact_path = os.path.join(MODELS_DIR, "logreg_artifacts.joblib")
    save_model_artifacts(logreg_artifact_path,
                        model=logreg, scaler=scaler, encoder=le_gender, feature_cols=feature_cols)

    # 5) XGBoost safe
    print("Training XGBoost classifier (safe mode)...")
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, neg/pos) if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        scale_pos_weight=scale_pos_weight
    )

    trained_model = None
    xgb_out_prefix = os.path.join(REPORT_DIR, "xgboost")
    try:
        xgb.fit(X_train, y_train, verbose=False)
        trained_model = xgb
        print("XGBoost trained successfully.")
    except Exception as e:
        print("Warning: XGBoost training failed:", e)
        print("Falling back to RandomForestClassifier...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        trained_model = rf
        xgb_out_prefix = os.path.join(REPORT_DIR, "random_forest_fallback")

    model_name = type(trained_model).__name__
    model_metrics = evaluate_and_save(model_name, trained_model, X_test, y_test, xgb_out_prefix)

    artifact_name = "xgb_artifacts.joblib" if isinstance(trained_model, XGBClassifier) else "rf_artifacts.joblib"
    artifact_path = os.path.join(MODELS_DIR, artifact_name)
    save_model_artifacts(artifact_path,
                        model=trained_model, scaler=scaler, encoder=le_gender, feature_cols=feature_cols)

    # 6) Feature importance plot (fixed)
    try:
        fmap = {}
        if hasattr(trained_model, "get_booster"):
            booster_scores = trained_model.get_booster().get_score(importance_type="gain")
            for k, v in booster_scores.items():
                if re.fullmatch(r"f\d+", k):
                    idx = int(k[1:])
                    fmap[feature_cols[idx]] = v
                else:
                    fmap[k] = v
            items = sorted(fmap.items(), key=lambda x: x[1], reverse=True)
            feat_names = [i[0] for i in items]
            feat_scores = [i[1] for i in items]
        elif hasattr(trained_model, "feature_importances_"):
            feat_names = feature_cols
            feat_scores = trained_model.feature_importances_.tolist()
            items = sorted(zip(feat_names, feat_scores), key=lambda x: x[1], reverse=True)
            feat_names = [i[0] for i in items]
            feat_scores = [i[1] for i in items]
        else:
            feat_names, feat_scores = [], []

        if feat_names:
            plt.figure(figsize=(8, max(4, len(feat_names)*0.3)))
            y_pos = np.arange(len(feat_names))
            plt.barh(y_pos, feat_scores[::-1])
            plt.yticks(y_pos, feat_names[::-1])
            plt.title(f"Feature Importance ({model_name})")
            plt.xlabel("Importance")
            plt.tight_layout()
            feat_plot_path = os.path.join(REPORT_DIR, f"{model_name.lower()}_feature_importance.png")
            plt.savefig(feat_plot_path)
            plt.clf()
            print(f"Saved feature importance plot to {feat_plot_path}")
    except Exception as e:
        print("Warning: Could not produce feature importance plot:", e)

    # 7) Save model summary CSV
    summary_csv_path = os.path.join(REPORT_DIR, "model_summary.csv")
    summary_df = pd.DataFrame([{
        "model": "LogisticRegression",
        "accuracy": lr_metrics["accuracy"],
        "roc_auc": lr_metrics["roc_auc"]
    }, {
        "model": model_name,
        "accuracy": model_metrics["accuracy"],
        "roc_auc": model_metrics["roc_auc"]
    }])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved model summary to {summary_csv_path}")

    print("All done. Artifacts saved in:", MODELS_DIR, "and reports in", REPORT_DIR)


if __name__ == "__main__":
    main()
