#!/usr/bin/env python3
"""
Hyperparameter tuning for the crypto volatility prediction model
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURES_FILE = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
TUNED_MODEL_FILE = MODEL_DIR / "tuned_model.pkl"
METRICS_FILE = MODEL_DIR / "tuned_metrics.json"
FEATURE_IMPORTANCE_FILE = MODEL_DIR / "feature_importance.json"

FEATURE_COLS = [
    "return_pct", "ret_std_3d", 
    "ma_5d", "ma_10d", "rsi_14d",
    "volatility_1d", "volatility_3d", "volatility_7d",
    "bb_width", "bb_pos",
    "mom_5d", "mom_10d",
    "macd", "macd_signal", "macd_diff",
    "realized_vol_5d", "realized_vol_10d", "realized_vol_30d",
    "ewm_vol_5d", "ewm_vol_10d", "ewm_vol_30d", 
    "vol_of_vol_10d", "parkinson_vol"
]

def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")

def main() -> None:
    if not FEATURES_FILE.exists():
        ts_log("ERROR", "features.csv not found. Cannot tune model.")
        raise SystemExit(1)

    ts_log("INFO", f"Loading feature data from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    
    if df.empty:
        ts_log("ERROR", "Features file is empty. Cannot tune model.")
        raise SystemExit(1)
    ts_log("INFO", f"Loaded feature data with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Check for and drop columns that are all NaN
    nan_cols = df.columns[df.isna().all()].tolist()
    if nan_cols:
        ts_log("INFO", f"Dropping {len(nan_cols)} columns with all NaN values: {nan_cols}")
        df = df.drop(columns=nan_cols)
    
    
    if not df.empty:
        nan_percent = df.isna().mean()
        high_nan_cols = nan_percent[nan_percent > 0.5].index.tolist()
        if high_nan_cols:
            ts_log("INFO", f"Dropping {len(high_nan_cols)} columns with >50% NaN values: {high_nan_cols}")
            df = df.drop(columns=high_nan_cols)
    
    if df.empty:
        ts_log("ERROR", "Features file became empty after dropping all-NaN or high-NaN columns.")
        raise SystemExit(1)
    required_cols_for_processing = FEATURE_COLS + ['target', 'coin', 'date']
    
    # Filter this list to columns actually present in the DataFrame at this stage
    present_cols_for_subset = [col for col in required_cols_for_processing if col in df.columns]
    
    # Ensure critical columns like 'target' and at least some features are present
    if 'target' not in present_cols_for_subset:
        ts_log("ERROR", "Target column 'target' is missing from the DataFrame after initial NaN column drops.")
        raise SystemExit(1)
    
    present_feature_cols = [col for col in FEATURE_COLS if col in present_cols_for_subset]
    if not present_feature_cols:
        ts_log("ERROR", "None of the defined FEATURE_COLS are present in the DataFrame.")
        raise SystemExit(1)

    df_subset = df[present_cols_for_subset].copy()
    
    # Columns that must not have NaNs for a row to be kept
    dropna_on_these_cols = present_feature_cols + ['target']
        
    # Handle remaining missing values by dropping rows where essential modeling columns are NaN
    df_subset.dropna(subset=dropna_on_these_cols, inplace=True)

    if df_subset.empty:
        ts_log("ERROR", "No valid data after selecting modeling columns and dropping NaN values from them.")
        raise SystemExit(1)
        
    if "target" not in df_subset.columns:
        ts_log("ERROR", "Target column 'target' not found in features file subset.")
        raise SystemExit(1)
        
    class_counts = df_subset["target"].value_counts()
    ts_log("INFO", f"Class distribution: {class_counts.to_dict()}")
    
    if len(df_subset) < 30:
        ts_log("WARN", f"Only {len(df_subset)} samples available. This may be insufficient for reliable cross-validation.")

    numeric_features = df_subset[present_feature_cols].select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_features:
        ts_log("ERROR", "No numeric features found in the dataset from the selected FEATURE_COLS.")
        raise SystemExit(1)
    
    X = df_subset[numeric_features]
    y = df_subset["target"].astype(int)
    
    ts_log("INFO", f"Using {len(numeric_features)} numeric features for modeling: {numeric_features[:5]}...")
    
    if X.shape[0] == 0:
        ts_log("ERROR", "Empty feature matrix. Check the data processing.")
        raise SystemExit(1)
        
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        ts_log("INFO", f"Removing {len(constant_features)} constant features: {constant_features}")
        X = X.drop(columns=constant_features)
        
    if X.shape[1] == 0:
        ts_log("ERROR", "No non-constant features remaining for modeling.")
        raise SystemExit(1)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    class_counts = Counter(y)
    neg, pos = class_counts[0], class_counts[1]
    base_imbalance_ratio = neg / pos if pos > 0 else 1 # calculate base ratio
    # Cap the imbalance ratio to avoid extreme scale_pos_weight values
    capped_imbalance_ratio = min(base_imbalance_ratio, 4.0) 

    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        # Adjusted scale_pos_weight to be more conservative
        'scale_pos_weight' : sorted(list(set([1, round(capped_imbalance_ratio * 0.75, 2), round(capped_imbalance_ratio, 2)])))
    }
    
    if len(X) < 50:
        ts_log("INFO", "Using smaller number of CV splits due to limited data.")
        n_splits_cv = 3
        test_size_cv = min(10, len(X) // (n_splits_cv + 1)) # Ensure test_size is reasonable for train set size
        test_size_cv = max(1, test_size_cv) # Ensure test_size is at least 1
    else:
        n_splits_cv = 5
        # Dynamic test_size: min 14 days, max 30 days, or 1/6th of data if smaller
        test_size_cv = max(14, min(30, len(X) // (n_splits_cv + 1))) 
        
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    cv = TimeSeriesSplit(n_splits=n_splits_cv, test_size=test_size_cv)

    ts_log("INFO", f"Using time-series aware cross-validation with {n_splits_cv} splits and test_size {test_size_cv}.")
    
    ts_log("INFO", "Starting hyperparameter tuning (this may take a while)...")
    
    # Adjust n_iter for RandomizedSearchCV based on available data and CV setup
    n_iter = min(20, len(X) // (n_splits_cv * 2)) # Ensure enough samples per iter, e.g. at least 2x test_size per training fold implies n_splits * test_size * 2 minimum data
    n_iter = max(5, n_iter) # Minimum 5 iterations
    ts_log("INFO", f"RandomizedSearchCV n_iter: {n_iter}")
    
    search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1 
    )
    
    try:
        search.fit(X_scaled, y)
    except Exception as e:
        ts_log("ERROR", f"Error during hyperparameter search: {str(e)}")
        raise SystemExit(1)
    
    best_params = search.best_params_
    ts_log("INFO", f"Best parameters: {best_params}")
    
    # Save best_params to a JSON file
    BEST_XGB_PARAMS_FILE = MODEL_DIR / "best_xgb_params.json"
    with BEST_XGB_PARAMS_FILE.open("w") as fp:
        json.dump(best_params, fp, indent=2)
    ts_log("INFO", f"Saved best XGBoost parameters to {BEST_XGB_PARAMS_FILE.relative_to(Path('.'))}")
    
    best_model = xgb.XGBClassifier(**best_params, random_state=42)
    
    # 5-fold cross-validation with best model
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    feature_importances_sum = {feature: 0.0 for feature in numeric_features} # Ensure float for summation
    
    for train_idx, test_idx in cv.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        best_model.fit(X_train, y_train)
        
        # Update feature importances
        for feature, importance in zip(numeric_features, best_model.feature_importances_):
            feature_importances_sum[feature] += float(importance) # Sum raw importances
        
        # Get predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        scores["roc_auc"].append(roc_auc_score(y_test, y_prob))
    
    # Calculate mean metrics
    mean_metrics = {
        "mean_accuracy": sum(scores["accuracy"]) / 5,
        "mean_precision": sum(scores["precision"]) / 5,
        "mean_recall": sum(scores["recall"]) / 5,
        "mean_f1": sum(scores["f1"]) / 5,
        "mean_roc_auc": sum(scores["roc_auc"]) / 5,
    }
    
    ts_log(
        "INFO",
        "Cross-Validation Metrics (Mean): "
        f"Accuracy={mean_metrics['mean_accuracy']:.4f}, "
        f"Precision={mean_metrics['mean_precision']:.4f}, "
        f"Recall={mean_metrics['mean_recall']:.4f}, "
        f"F1={mean_metrics['mean_f1']:.4f}, "
        f"ROC AUC={mean_metrics['mean_roc_auc']:.4f}"
    )
    
    # Save metrics
    with METRICS_FILE.open("w") as fp:
        json.dump(mean_metrics, fp, indent=2)
    ts_log("INFO", f"Saved tuned metrics to {METRICS_FILE.relative_to(Path('.'))}")
    
    
    num_cv_folds = cv.get_n_splits()

    # Convert summed importances to average
    averaged_importances = {name: total_imp / num_cv_folds for name, total_imp in feature_importances_sum.items()}
    
    sorted_importances = dict(sorted(averaged_importances.items(), key=lambda x: x[1], reverse=True))
    
    with FEATURE_IMPORTANCE_FILE.open("w") as fp:
        json_safe_importances = {k: float(v) for k, v in sorted_importances.items()}
        json.dump(json_safe_importances, fp, indent=2)
    ts_log("INFO", f"Saved averaged feature importances to {FEATURE_IMPORTANCE_FILE.relative_to(Path('.'))}")
    
    # Feature selection based on the averaged importances from the cross-validated best_model
    ts_log("INFO", "Performing feature selection based on averaged cross-validated importances...")
    
    importance_df_from_cv = pd.DataFrame(list(sorted_importances.items()), columns=['feature', 'importance'])
    
    # Keep only features with importance above mean
    mean_cv_importance = importance_df_from_cv['importance'].mean()
    important_features = importance_df_from_cv[importance_df_from_cv['importance'] > mean_cv_importance]['feature'].tolist()

    if not important_features:
        ts_log("WARN", "No features selected based on mean importance threshold.")
        if not importance_df_from_cv.empty:
            num_top_features = min(5, len(importance_df_from_cv))
            if num_top_features > 0:
                important_features = importance_df_from_cv['feature'].tolist()[:num_top_features]
                ts_log("WARN", f"Using top {len(important_features)} features as a fallback: {important_features}")
            else:
                important_features = X_scaled.columns.tolist()
                ts_log("WARN", "Fallback to using all features as no top features could be determined.")
        else:
            important_features = X_scaled.columns.tolist()
            ts_log("WARN", "Fallback to using all features as importance_df was empty.")


    ts_log("INFO", f"Selected {len(important_features)} important features out of {len(X_scaled.columns)}: {important_features[:5]}")

    X_important = X_scaled[important_features]
    
    ts_log("INFO", "Training final model with tuned parameters on selected important features...")
    final_model = xgb.XGBClassifier(**best_params, random_state=42)
    final_model.fit(X_important, y)
    
    with TUNED_MODEL_FILE.open("wb") as fp:
        pickle.dump((final_model, scaler, important_features), fp)
    ts_log("INFO", f"Saved tuned model to {TUNED_MODEL_FILE.relative_to(Path('.'))}")

if __name__ == "__main__":
    main()