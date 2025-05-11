# File: run_advanced_pipeline.sh (New file)
#!/bin/bash
set -e  # Exit on error

echo "== Starting crypto volatility production pipeline =="

echo "1. Fetching coin data..."
python -m crypto_vol.scripts.fetch_data

# echo "2. Fetching market context data..."
# python -m crypto_vol.scripts.fetch_market_data

echo "2. Preprocessing data..."
python -m crypto_vol.scripts.preprocess

echo "3. Tuning XGBoost model..."
python -m crypto_vol.scripts.tune_model

# echo "5. Building ensemble model..."
# python -m crypto_vol.scripts.ensemble_model

echo "4. Making predictions with tuned XGBoost model..."
python -m crypto_vol.scripts.predict

# echo "7. Making predictions with ensemble model..."
# python -m crypto_vol.scripts.predict_ensemble

echo "5. Generating plots..."
python -m crypto_vol.scripts.generate_plots

echo "Production pipeline completed successfully!"