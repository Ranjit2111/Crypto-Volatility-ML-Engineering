#!/bin/bash
# Shell script wrapper - run full Phase-2 pipeline, build plots

# Run pipeline scripts as modules
python -m crypto_vol.scripts.fetch_data
python -m crypto_vol.scripts.preprocess
python -m crypto_vol.scripts.train
python -m crypto_vol.scripts.predict
python -m crypto_vol.scripts.generate_plots