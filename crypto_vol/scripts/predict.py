#!/usr/bin/env python3
"""
Generate next‑day volatility probabilities (Phase 2 – XGBoost, 10 coins).
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_FILE = Path("models/model.pkl")
FEATURES_FILE = Path("data/processed/features.csv")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
FEATURE_COLS = ["return_pct", "ret_std_3d", "ma_5d", "ma_10d", "rsi_14d"]


def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")


def main() -> None:
    if not MODEL_FILE.exists():
        ts_log("ERROR", "Model file not found. Run training first.")
        raise SystemExit(1)
    if not FEATURES_FILE.exists():
        ts_log("ERROR", "features.csv not found. Run preprocessing first.")
        raise SystemExit(1)

    ts_log("INFO", f"Loading Phase 2 model from {MODEL_FILE}...")
    with MODEL_FILE.open("rb") as fp:
        model = pickle.load(fp)

    ts_log("INFO", "Loading latest feature data...")
    df = pd.read_csv(FEATURES_FILE)
    latest_per_coin = (
        df.sort_values("date")
        .groupby("coin")
        .tail(1)
        .reset_index(drop=True)
    )
    X_latest = latest_per_coin[FEATURE_COLS]
    probs = model.predict_proba(X_latest)[:, 1]

    predictions_list: list[dict[str, Any]] = []
    for coin, prob in zip(latest_per_coin["coin"], probs):
        predictions_list.append(
            {"coin": coin, "volatility_probability": round(float(prob), 6)}
        )

    predictions_list.sort(key=lambda x: x["volatility_probability"], reverse=True)
    most_volatile_coin = predictions_list[0]["coin"]

    output = {
        "prediction_date": TODAY,
        "most_volatile_coin": most_volatile_coin,
        "predictions": predictions_list,
    }

    out_file = OUTPUT_DIR / f"predictions_{TODAY}.json"
    with out_file.open("w") as fp:
        json.dump(output, fp, indent=2)

    ts_log("INFO", f"Predicted most volatile coin: {most_volatile_coin}")
    ts_log("INFO", f"Saving predictions to {out_file.relative_to(Path('.'))}...")


if __name__ == "__main__":
    main()