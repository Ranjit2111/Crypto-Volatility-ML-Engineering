#!/usr/bin/env python3
"""
Phase 2 model training:
* Features: return_pct + engineered features
* Model  : XGBoost classifier
* 5‑fold CV &rarr; metrics.json
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold

FEATURES_FILE = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"

FEATURE_COLS = ["return_pct", "ret_std_3d", "ma_5d", "ma_10d", "rsi_14d"]


def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")


def main() -> None:
    if not FEATURES_FILE.exists():
        ts_log("ERROR", "features.csv not found. Cannot train model.")
        raise SystemExit(1)

    ts_log("INFO", f"Loading feature data from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)

    X = df[FEATURE_COLS]
    y = df["target"].astype(int)

    model_params = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
    )

    # 5‑fold cross‑validation
    ts_log("INFO", "Performing 5‑fold cross‑validation (XGBoost)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = xgb.XGBClassifier(**model_params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))

    mean_metrics = {
        "mean_accuracy": sum(scores["accuracy"]) / 5,
        "mean_precision": sum(scores["precision"]) / 5,
        "mean_recall": sum(scores["recall"]) / 5,
        "mean_f1": sum(scores["f1"]) / 5,
    }
    ts_log(
        "INFO",
        "Cross-Validation Metrics (Mean): "
        f"Accuracy={mean_metrics['mean_accuracy']:.4f}, "
        f"Precision={mean_metrics['mean_precision']:.4f}, "
        f"Recall={mean_metrics['mean_recall']:.4f}, "
        f"F1={mean_metrics['mean_f1']:.4f}",
    )

    with METRICS_FILE.open("w") as fp:
        json.dump(mean_metrics, fp, indent=2)
    ts_log("INFO", f"Saved CV metrics to {METRICS_FILE.relative_to(Path('.'))}")

    # Train final model on all data
    ts_log("INFO", "Training final XGBoost model on full dataset...")
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X, y)

    with MODEL_FILE.open("wb") as fp:
        pickle.dump(final_model, fp)
    ts_log("INFO", f"Saved Phase 2 model to {MODEL_FILE.relative_to(Path('.'))}")


if __name__ == "__main__":
    main()