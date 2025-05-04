#!/bin/bash
set -e

python3 scripts/fetch_data.py
python3 scripts/preprocess.py
python3 scripts/train.py
python3 scripts/predict.py