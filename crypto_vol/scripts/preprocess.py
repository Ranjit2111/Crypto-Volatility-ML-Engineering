#!/usr/bin/env python3
"""
Preprocess raw data into features.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import ta

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")


def load_raw_today() -> pd.DataFrame:

    files = sorted(RAW_DIR.glob("*.json"))
    ts_log("INFO", f"Found {len(files)} raw data files")
    if not files:
        ts_log("WARN", "No raw data files found. Exiting.")
        raise SystemExit(0)
    
    dates = set()
    for f in files:
        parts = f.name.split('_')
        if len(parts) >= 2:
            date_part = parts[1].replace('.json', '')
            dates.add(date_part)
    
    if not dates:
        ts_log("WARN", "Couldn't extract dates from filenames. Exiting.")
        raise SystemExit(0)
    
    latest_date = sorted(dates)[-1]
    ts_log("INFO", f"Using data files from {latest_date}")
    
    latest_files = [f for f in files if f.name.endswith(f"_{latest_date}.json")]
    ts_log("INFO", f"Found {len(latest_files)} files for date {latest_date}")
    if not latest_files:
        ts_log("WARN", f"No files found for date {latest_date}. Exiting.")
        raise SystemExit(0)
    
    rows: list[pd.DataFrame] = []
    for f in latest_files:
        coin = f.name.split("_")[0]
        try:
            with f.open() as fp:
                raw = json.load(fp) #if no json data, then JSONDecoderError will show up
                
            # Print sample of raw data for debugging
            ts_log("DEBUG", f"Sample data for {coin}: {raw[:2] if raw else 'Empty'}")
                
            # Check if data is not empty
            if not raw:
                ts_log("WARN", f"Empty data for {coin}, skipping.")
                continue
             
            # The data is a list of lists with [timestamp_ms, open, high, low, close]
            # Convert to a dataframe with appropriate column names
            df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close"])
            
            # Ensure numeric types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df["coin"] = coin
            
            # Skip if all numeric data is NaN
            if df[["open", "high", "low", "close"]].isna().all().all():
                ts_log("WARN", f"All price data for {coin} is NaN, skipping.")
                continue
                
            rows.append(df)
            ts_log("DEBUG", f"Added {len(df)} rows for {coin}")
        except (json.JSONDecodeError, ValueError) as e:
            ts_log("ERROR", f"Failed to load data for {coin}: {e}")
            continue
        except Exception as e:
            ts_log("ERROR", f"Unexpected error processing {coin}: {str(e)}")
            continue

    if not rows:
        ts_log("ERROR", "No valid data could be loaded from any coin files. Exiting.")
        raise SystemExit(1)
        
    result = pd.concat(rows, ignore_index=True)
    ts_log("INFO", f"Total rows loaded: {result.shape[0]}..... Total columns loaded: {result.shape[1]}")
    return result

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified engineering function that creates all needed features."""
    ts_log("INFO", "Resampling hourly data to daily.")
    
    # Debug input
    ts_log("DEBUG", f"Engineer features input shape: {df.shape}")
    ts_log("DEBUG", f"Engineer features columns: {df.columns.tolist()}")
    
    # Make sure we have the required columns
    required_cols = ["timestamp_ms", "open", "high", "low", "close", "coin"]
    for col in required_cols:
        if col not in df.columns:
            ts_log("ERROR", f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")
    
    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    
    # Extract date for daily aggregation
    df["date"] = df["datetime"].dt.date
    
    # Aggregate to daily level
    daily_agg = df.groupby(["coin", "date"]).agg({
        "open": "first", 
        "high": "max", 
        "low": "min", 
        "close": "last"
    }).reset_index()
    
    # Create datetime column
    daily_agg["datetime"] = pd.to_datetime(daily_agg["date"])
    daily_agg["date_str"] = daily_agg["date"].astype(str)
    
    # Sort by coin and date for time-series calculations
    daily_agg = daily_agg.sort_values(["coin", "datetime"])
    
    # Basic return calculation
    daily_agg["return_pct"] = (daily_agg["close"] - daily_agg["open"]) / daily_agg["open"] * 100
    
    ts_log("INFO", "Calculating technical indicators...")
    
    # Process each coin separately to ensure proper time ordering
    result_dfs = []
    for coin, group in daily_agg.groupby("coin"):
        group = group.sort_values("datetime").copy()
        
        # 3-day rolling standard deviation of returns
        group["ret_std_3d"] = group["return_pct"].rolling(window=3).std()
        
        # Moving averages
        group["ma_5d"] = group["close"].rolling(window=5).mean()
        group["ma_10d"] = group["close"].rolling(window=10).mean()
        
        # RSI - Relative Strength Index
        delta = group["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        group["rsi_14d"] = 100 - (100 / (1 + rs))
        
        # Volatility
        group["volatility_1d"] = (group["high"] - group["low"]) / group["open"] * 100
        group["volatility_3d"] = group["volatility_1d"].rolling(window=3).mean()
        group["volatility_7d"] = group["volatility_1d"].rolling(window=7).mean()
        
        # Bollinger Bands
        group["bb_middle"] = group["close"].rolling(window=20).mean()
        bb_std = group["close"].rolling(window=20).std()
        group["bb_upper"] = group["bb_middle"] + 2 * bb_std
        group["bb_lower"] = group["bb_middle"] - 2 * bb_std
        group["bb_width"] = (group["bb_upper"] - group["bb_lower"]) / group["bb_middle"]
        group["bb_pos"] = (group["close"] - group["bb_lower"]) / (group["bb_upper"] - group["bb_lower"])
        
        # Momentum
        group["mom_5d"] = group["close"].pct_change(periods=5) * 100
        group["mom_10d"] = group["close"].pct_change(periods=10) * 100
        
        # MACD
        ema12 = group["close"].ewm(span=12, adjust=False).mean()
        ema26 = group["close"].ewm(span=26, adjust=False).mean()
        group["macd"] = ema12 - ema26
        group["macd_signal"] = group["macd"].ewm(span=9, adjust=False).mean()
        group["macd_diff"] = group["macd"] - group["macd_signal"]
        
        # Log returns for volatility calculations
        group["log_return"] = np.log(group["close"] / group["close"].shift(1))
        
        # Realized volatility
        group["realized_vol_5d"] = group["log_return"].rolling(window=5).std() * np.sqrt(365)
        group["realized_vol_10d"] = group["log_return"].rolling(window=10).std() * np.sqrt(365)
        group["realized_vol_30d"] = group["log_return"].rolling(window=30).std() * np.sqrt(365)
        
        # EWM volatility
        group["ewm_vol_5d"] = group["log_return"].ewm(span=5).std() * np.sqrt(365)
        group["ewm_vol_10d"] = group["log_return"].ewm(span=10).std() * np.sqrt(365)
        group["ewm_vol_30d"] = group["log_return"].ewm(span=30).std() * np.sqrt(365)
        
        # Volatility of volatility
        group["vol_of_vol_10d"] = group["realized_vol_5d"].rolling(window=10).std()
        
        # Parkinson volatility
        group["parkinson_vol"] = np.sqrt((1 / (4 * np.log(2))) * 
                                        np.log(group["high"] / group["low"]) ** 2) * np.sqrt(365)
        
        # Next day's volatility (for target creation)
        group["next_volatility"] = group["volatility_1d"].shift(-1)
        
        result_dfs.append(group)
    
    # Combine all processed coins
    processed = pd.concat(result_dfs)
    
    # Create target based on volatility ranking
    processed["volatility_rank"] = processed.groupby("coin")["next_volatility"].transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
    )
    
    # Top 20% volatility days (8,9 are the top 20%)
    processed["target"] = (processed["volatility_rank"] >= 8).astype("Int64")
    
    # Drop NaN rows
    n_before = len(processed)
    
    # Only drop rows with NaN in critical features, not all columns
    critical_cols = ["return_pct", "volatility_1d", "ma_5d"]
    processed = processed.dropna(subset=critical_cols)
    
    n_after = len(processed)
    
    ts_log("INFO", f"Dropped {n_before - n_after} rows with NaN values in critical columns")
    
    # For remaining NaN values, fill with appropriate defaults
    # Fill 0 for most features, mean for others
    processed = processed.fillna({
        "ret_std_3d": processed["ret_std_3d"].median(),  # realistic fallback
        "ma_10d": processed["ma_10d"].fillna(processed["ma_5d"]),  # 5d is best-effort
        "rsi_14d": 50,  # neutral RSI
        "volatility_3d": processed["volatility_3d"].median(),  # use past volatility median
        "volatility_7d": processed["volatility_7d"].median(),
        "bb_width": processed["bb_width"].median(),
        "bb_pos": 0.5,  # center — okay
        "mom_5d": 0.0,  # flat
        "mom_10d": 0.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_diff": 0.0,
        "realized_vol_5d": processed["realized_vol_5d"].median(),
        "realized_vol_10d": processed["realized_vol_10d"].median(),
        "realized_vol_30d": processed["realized_vol_30d"].median(),
        "ewm_vol_5d": processed["ewm_vol_5d"].median(),
        "ewm_vol_10d": processed["ewm_vol_10d"].median(),
        "ewm_vol_30d": processed["ewm_vol_30d"].median(),
        "vol_of_vol_10d": processed["vol_of_vol_10d"].median(),
        "parkinson_vol": processed["parkinson_vol"].median()
    })
        
    # Keep only the required columns
    final_cols = [
        "coin", "date_str", 
        "return_pct", "ret_std_3d", 
        "ma_5d", "ma_10d", "rsi_14d",
        "volatility_1d", "volatility_3d", "volatility_7d",
        "bb_width", "bb_pos",
        "mom_5d", "mom_10d",
        "macd", "macd_signal", "macd_diff",
        "realized_vol_5d", "realized_vol_10d", "realized_vol_30d",
        "ewm_vol_5d", "ewm_vol_10d", "ewm_vol_30d", 
        "vol_of_vol_10d", "parkinson_vol",
        "target"
    ]
    
    # Rename date_str to date for compatibility with rest of pipeline
    processed = processed[final_cols].rename(columns={"date_str": "date"})
    
    ts_log("DEBUG", f"Final processed shape: {processed.shape}")
    ts_log("DEBUG", f"Final columns: {processed.columns.tolist()}")
    
    return processed

def main() -> None:
    ts_log("INFO", f"Starting preprocessing for date {TODAY}...")
    raw_df = load_raw_today()
    
    # Debug raw data
    ts_log("INFO", f"Raw data shape: {raw_df.shape}")
    ts_log("INFO", f"Raw data columns: {raw_df.columns.tolist()}")
    ts_log("INFO", f"Loaded data for coins: {sorted(raw_df['coin'].unique())}")
    
    processed = engineer_features(raw_df)
    
    out_file = PROC_DIR / "features.csv"
    processed.to_csv(out_file, index=False)
    ts_log(
        "INFO",
        f"Processed data shape: {processed.shape}. Saved to {out_file.relative_to(Path('.'))}.",
    )

if __name__ == "__main__":
    main()