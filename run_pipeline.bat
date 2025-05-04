@echo off
SETLOCAL

REM Option 1: Run as module (recommended)
python -m crypto_vol.scripts.fetch_data
python -m crypto_vol.scripts.preprocess
python -m crypto_vol.scripts.train
python -m crypto_vol.scripts.predict
python -m crypto_vol.scripts.generate_plots

ENDLOCAL