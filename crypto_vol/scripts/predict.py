#!/usr/bin/env python3
"""
Predict the most volatile coin for the next day.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

MODEL_FILE = Path("models/tuned_model.pkl")
FEATURES_FILE = Path("data/processed/features.csv")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")


def main() -> None:
    if not MODEL_FILE.exists():
        ts_log("ERROR", "Tuned model file not found. Run tune_model.py first.")
        raise SystemExit(1)
    if not FEATURES_FILE.exists():
        ts_log("ERROR", "features.csv not found. Run preprocessing first.")
        raise SystemExit(1)

    ts_log("INFO", f"Loading improved model from {MODEL_FILE}...")
    with MODEL_FILE.open("rb") as fp:
        model, scaler, important_features = pickle.load(fp)

    ts_log("INFO", f"Model uses {len(important_features)} selected features for prediction")
    ts_log("INFO", "Loading latest feature data...")
    latest_df = pd.read_csv(FEATURES_FILE)
    latest_df = latest_df.sort_values(["coin", "date"])
    
    # Handle NaN columns (like market data)
    nan_cols = latest_df.columns[latest_df.isna().all()].tolist()
    if nan_cols:
        ts_log("INFO", f"Dropping {len(nan_cols)} columns with all NaN values from latest data: {nan_cols}")
        latest_df = latest_df.drop(columns=nan_cols)
        
    # Select only the most recent data point for each coin
    latest_df_per_coin = latest_df.groupby("coin").tail(1).copy() # Use a copy

    # Get the list of features the scaler was fit on (from the scaler itself)
    scaler_features = scaler.feature_names_in_.tolist()
    
    # Ensure these features are in the latest data
    missing_scaler_features = [f for f in scaler_features if f not in latest_df_per_coin.columns]
    if missing_scaler_features:
        ts_log("ERROR", f"Features the scaler was fit on are missing from latest data: {missing_scaler_features}")
        # Fill with 0 as a fallback - this might not be ideal for all features
        for col in missing_scaler_features:
            latest_df_per_coin[col] = 0
            
    # Prepare the data for scaling (only columns scaler expects)
    X_for_scaling = latest_df_per_coin[scaler_features].copy()
    
    # Fill any NaNs in X_for_scaling before scaling
    if X_for_scaling.isna().any().any():
        ts_log("WARN", "NaN values found in data for scaling, filling with 0.")
        X_for_scaling = X_for_scaling.fillna(0)
        
    if X_for_scaling.empty:
        ts_log("ERROR", "No data available for scaling after filtering.")
        predictions_df = pd.DataFrame(columns=["coin", "date", "prob_volatility_next_day"])
    else:
        # Scale the data (all features the scaler was fit on)
        X_scaled_all_features = pd.DataFrame(
            scaler.transform(X_for_scaling), 
            columns=X_for_scaling.columns, 
            index=X_for_scaling.index
        )
        
        # Now select the important features from the scaled data
        missing_important_from_scaled = [f for f in important_features if f not in X_scaled_all_features.columns]
        if missing_important_from_scaled:
            ts_log("ERROR", f"Important features are missing after scaling: {missing_important_from_scaled}")
            # This shouldn't happen if scaler_features contained important_features
            # As a fallback, add missing columns with 0
            for col in missing_important_from_scaled:
                 X_scaled_all_features[col] = 0
                 
        X_latest_scaled_important = X_scaled_all_features[important_features]

        # Make predictions
        ts_log("INFO", "Making predictions...")
        probabilities = model.predict_proba(X_latest_scaled_important)[:, 1]
        
        # Prepare output
        predictions_df = pd.DataFrame({
            "coin": latest_df_per_coin["coin"],
            "date": latest_df_per_coin["date"],
            "prob_volatility_next_day": probabilities,
        })

    # For logging confidence
    confidence_stats = {
        "mean": float(np.mean(probabilities)),
        "min": float(np.min(probabilities)),
        "max": float(np.max(probabilities)),
        "std": float(np.std(probabilities))
    }
    ts_log("INFO", f"Prediction confidence stats: {confidence_stats}")

    predictions_list: list[dict[str, Any]] = []
    for coin, prob in zip(latest_df_per_coin["coin"], probabilities):
        predictions_list.append(
            {"coin": coin, "volatility_probability": round(float(prob), 6)}
        )

    predictions_list.sort(key=lambda x: x["volatility_probability"], reverse=True)
    most_volatile_coin = predictions_list[0]["coin"]

    output = {
        "prediction_date": TODAY,
        "most_volatile_coin": most_volatile_coin,
        "predictions": predictions_list,
        "confidence_stats": confidence_stats,
        "model_type": "xgboost"
    }

    out_file = OUTPUT_DIR / f"predictions_{TODAY}.json"
    with out_file.open("w") as fp:
        json.dump(output, fp, indent=2)

    ts_log("INFO", f"Predicted most volatile coin: {most_volatile_coin}")
    ts_log("INFO", f"Saving predictions to {out_file.relative_to(Path('.'))}...")

if __name__ == "__main__":
    main()