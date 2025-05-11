@echo off
SETLOCAL

echo == Starting advanced crypto volatility pipeline ==

echo 1. Fetching coin data...
python -m crypto_vol.scripts.fetch_data

echo 2. Fetching market context data...
python -m crypto_vol.scripts.fetch_market_data

echo 3. Preprocessing data with advanced features...
python -m crypto_vol.scripts.preprocess

echo 4. Tuning model with feature selection...
python -m crypto_vol.scripts.tune_model

echo 5. Building ensemble model...
python -m crypto_vol.scripts.ensemble_model

echo 6. Making predictions with tuned model...
python -m crypto_vol.scripts.predict

echo 7. Making predictions with ensemble model...
python -m crypto_vol.scripts.predict_ensemble

echo 8. Generating plots...
python -m crypto_vol.scripts.generate_plots

echo Advanced pipeline completed successfully!

ENDLOCAL