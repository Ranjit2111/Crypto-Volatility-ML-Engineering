#!/bin/bash
# Shell script wrapper - run full Phase-2 pipeline, build plots

# Run pipeline scripts
python scripts/fetch_data.py
python scripts/preprocess.py
python scripts/train.py
python scripts/predict.py
python scripts/generate_plots.py 