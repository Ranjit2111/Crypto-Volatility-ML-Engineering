@echo off
SETLOCAL

echo == Starting crypto volatility production pipeline ==

echo 1. Fetching coin data...
python -m crypto_vol.scripts.fetch_data

echo 2. Preprocessing data...
python -m crypto_vol.scripts.preprocess

echo 3. Tuning XGBoost model...
python -m crypto_vol.scripts.tune_model

echo 4. Making predictions with tuned XGBoost model...
python -m crypto_vol.scripts.predict

echo 5. Generating plots...
python -m crypto_vol.scripts.generate_plots

echo Production pipeline completed successfully!

ENDLOCAL